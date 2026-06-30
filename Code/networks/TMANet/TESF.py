from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


DIM_TEXT_EMB = 512
DEFAULT_EDGE_OPERATOR_MODE = "sobel"
DEFAULT_TEXT_MODULATION_MODE = "film"
SUPPORTED_EDGE_OPERATOR_MODES = ("sobel", "prewitt", "laplacian", "roberts")
SUPPORTED_TEXT_MODULATION_MODES = ("film", "cross_attention")


def _num_groups(channels: int, max_groups: int = 16) -> int:
    """Choose a GroupNorm group count that always divides channels."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "trilinear") -> torch.Tensor:
    """Resize a 5D tensor to the spatial size of another 5D tensor."""
    if x.shape[2:] == ref.shape[2:]:
        return x
    if mode == "nearest":
        return F.interpolate(x, size=ref.shape[2:], mode=mode)
    return F.interpolate(x, size=ref.shape[2:], mode=mode, align_corners=False)


def _safe_repeat_text(text_embedding: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Accept (1, 512) or (B, 512) text embeddings and match the image batch."""
    if text_embedding.dim() != 2:
        raise ValueError("text_embedding must have shape (1, 512) or (B, 512).")
    if text_embedding.size(0) == batch_size:
        return text_embedding
    if text_embedding.size(0) == 1:
        return text_embedding.repeat(batch_size, 1)
    raise ValueError("text_embedding batch dimension does not match feature batch size.")


class CutBoundaryExtractor3D(nn.Module):
    """
    Step 1a: extract the explicit cut boundary from a mix mask.

    The module uses 3D dilation and erosion to create a narrow boundary band:
        E_cut = dilate(mask) - erode(mask)

    Args:
        band_width: odd kernel size for the boundary band. Larger values cover a
            thicker context around the mix boundary.
    """

    def __init__(self, band_width: int = 3):
        super().__init__()
        if band_width < 3 or band_width % 2 == 0:
            raise ValueError("band_width must be an odd integer >= 3.")
        self.band_width = band_width
        self.padding = band_width // 2

    def forward(self, mix_mask: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if mix_mask is None:
            return ref.new_zeros((ref.size(0), 1, *ref.shape[2:]))

        if mix_mask.dim() != 5:
            raise ValueError("mix_mask must be a 5D tensor: (B, 1, H, W, D).")

        mask = mix_mask
        if mask.size(1) != 1:
            mask = mask[:, :1]
        mask = mask.to(dtype=ref.dtype, device=ref.device)
        mask = _resize_like(mask, ref, mode="nearest")
        mask = (mask > 0.5).to(dtype=ref.dtype)

        dilated = F.max_pool3d(mask, kernel_size=self.band_width, stride=1, padding=self.padding)
        eroded = -F.max_pool3d(-mask, kernel_size=self.band_width, stride=1, padding=self.padding)
        return (dilated - eroded).clamp_(0.0, 1.0)


class ImageEdgeExtractor3D(nn.Module):
    """
    Step 1b: extract high-frequency image edges with fixed 3D filters.

    The edge operator is selected by operator_mode:
        - "sobel": 3D Sobel directional gradient magnitude.
        - "prewitt": 3D Prewitt directional gradient magnitude.
        - "laplacian": 6-neighbor 3D Laplacian absolute response.
        - "roberts": 3D Roberts-style diagonal finite differences.

    All modes work for anisotropic volumes because they never assume H, W, and
    D are equal; all spatial sizes are preserved.
    """

    def __init__(self, in_channels: int = 1, operator_mode: str = DEFAULT_EDGE_OPERATOR_MODE):
        super().__init__()
        if operator_mode not in SUPPORTED_EDGE_OPERATOR_MODES:
            raise ValueError(
                f"operator_mode must be one of {SUPPORTED_EDGE_OPERATOR_MODES}, got {operator_mode}."
            )
        self.in_channels = in_channels
        self.operator_mode = operator_mode
        kernels = self._build_kernels(operator_mode)
        if kernels is not None:
            self.register_buffer("kernels", kernels, persistent=False)
        else:
            self.kernels = None

    @staticmethod
    def _build_directional_kernels(smooth: torch.Tensor, deriv: torch.Tensor) -> torch.Tensor:
        k_h = deriv[:, None, None] * smooth[None, :, None] * smooth[None, None, :]
        k_w = smooth[:, None, None] * deriv[None, :, None] * smooth[None, None, :]
        k_d = smooth[:, None, None] * smooth[None, :, None] * deriv[None, None, :]
        kernels = torch.stack([k_h, k_w, k_d], dim=0).unsqueeze(1)
        return kernels / kernels.abs().sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _build_laplacian_kernel() -> torch.Tensor:
        kernel = torch.zeros(3, 3, 3)
        kernel[1, 1, 1] = -6.0
        kernel[0, 1, 1] = 1.0
        kernel[2, 1, 1] = 1.0
        kernel[1, 0, 1] = 1.0
        kernel[1, 2, 1] = 1.0
        kernel[1, 1, 0] = 1.0
        kernel[1, 1, 2] = 1.0
        return kernel.view(1, 1, 3, 3, 3)

    @classmethod
    def _build_kernels(cls, operator_mode: str) -> Optional[torch.Tensor]:
        deriv = torch.tensor([-1.0, 0.0, 1.0])
        if operator_mode == "sobel":
            return cls._build_directional_kernels(torch.tensor([1.0, 2.0, 1.0]), deriv)
        if operator_mode == "prewitt":
            return cls._build_directional_kernels(torch.tensor([1.0, 1.0, 1.0]), deriv)
        if operator_mode == "laplacian":
            return cls._build_laplacian_kernel()
        if operator_mode == "roberts":
            return None
        raise ValueError(f"Unsupported operator_mode: {operator_mode}")

    @staticmethod
    def _shift(x: torch.Tensor, offsets: Tuple[int, int, int]) -> torch.Tensor:
        pad = []
        slices = [slice(None), slice(None)]
        for dim, offset in zip((4, 3, 2), offsets[::-1]):
            if offset > 0:
                pad.extend([offset, 0])
                slices.insert(2, slice(0, -offset))
            elif offset < 0:
                pad.extend([0, -offset])
                slices.insert(2, slice(-offset, None))
            else:
                pad.extend([0, 0])
                slices.insert(2, slice(None))
        return F.pad(x[tuple(slices)], pad)

    @classmethod
    def _roberts_edge(cls, gray: torch.Tensor) -> torch.Tensor:
        # 3D Roberts-style diagonal differences across a 2x2x2 local cube.
        diagonal_pairs = (
            ((1, 1, 1), (0, 0, 0)),
            ((1, 1, 0), (0, 0, 1)),
            ((1, 0, 1), (0, 1, 0)),
            ((0, 1, 1), (1, 0, 0)),
        )
        responses = []
        for plus_offset, minus_offset in diagonal_pairs:
            plus = cls._shift(gray, plus_offset)
            minus = cls._shift(gray, minus_offset)
            responses.append((plus - minus).pow(2))
        return torch.sqrt(torch.stack(responses, dim=0).sum(dim=0) + 1e-6)

    def forward(self, image: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if image.dim() != 5:
            raise ValueError("image must be a 5D tensor: (B, C, H, W, D).")

        image = image.to(dtype=ref.dtype, device=ref.device)
        image = _resize_like(image, ref)

        # Average input modalities before fixed edge filtering. This keeps the
        # edge prior single-channel and independent of the image channel count.
        gray = image.mean(dim=1, keepdim=True)
        if self.operator_mode == "roberts":
            edge = self._roberts_edge(gray)
        elif self.operator_mode == "laplacian":
            kernels = self.kernels.to(dtype=gray.dtype, device=gray.device)
            edge = torch.abs(F.conv3d(gray, kernels, padding=1))
        else:
            kernels = self.kernels.to(dtype=gray.dtype, device=gray.device)
            gradients = F.conv3d(gray, kernels, padding=1)
            edge = torch.sqrt((gradients * gradients).sum(dim=1, keepdim=True) + 1e-6)
        edge_min = edge.amin(dim=(2, 3, 4), keepdim=True)
        edge_max = edge.amax(dim=(2, 3, 4), keepdim=True)
        return (edge - edge_min) / (edge_max - edge_min).clamp_min(1e-6)


class BoundaryPriorFusion3D(nn.Module):
    """
    Step 1: combine explicit mix boundary and image high-frequency edge prior.

    Output:
        edge_prior: (B, 1, Hf, Wf, Df), aligned with the provided feature map.
    """

    def __init__(
        self,
        image_channels: int = 1,
        band_width: int = 3,
        operator_mode: str = DEFAULT_EDGE_OPERATOR_MODE,
    ):
        super().__init__()
        self.cut_boundary = CutBoundaryExtractor3D(band_width=band_width)
        self.image_edge = ImageEdgeExtractor3D(
            in_channels=image_channels,
            operator_mode=operator_mode,
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(8), 8),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        image: torch.Tensor,
        feature: torch.Tensor,
        mix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cut_edge = self.cut_boundary(mix_mask, feature)
        image_edge = self.image_edge(image, feature)
        return self.fuse(torch.cat([cut_edge, image_edge], dim=1))


class BoundaryFeatureExtractor3D(nn.Module):
    """
    Step 2: restrict image features to the detected boundary area.

    The edge prior is a soft attention map. Multiplication keeps the feature
    shape unchanged: (B, C, H, W, D).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    def forward(self, feature: torch.Tensor, edge_prior: torch.Tensor) -> torch.Tensor:
        edge_prior = _resize_like(edge_prior, feature)
        return self.refine(feature * edge_prior)


class TextBoundaryModulator3D(nn.Module):
    """
    Step 3: use text embedding to modulate boundary features.

    modulation_mode controls how text affects the edge feature:
        - "film": project text to channel-wise gamma/beta parameters.
        - "cross_attention": flatten edge features into Q tokens and use
          projected text tokens as K/V:
              F_edge_text = CrossAttention(Q=edge_tokens, K=T, V=T)
    """

    def __init__(
        self,
        channels: int,
        text_dim: int = DIM_TEXT_EMB,
        reduction: int = 4,
        modulation_mode: str = DEFAULT_TEXT_MODULATION_MODE,
        num_text_tokens: int = 4,
    ):
        super().__init__()
        if modulation_mode not in SUPPORTED_TEXT_MODULATION_MODES:
            raise ValueError(
                f"modulation_mode must be one of {SUPPORTED_TEXT_MODULATION_MODES}, got {modulation_mode}."
            )
        if num_text_tokens < 1:
            raise ValueError("num_text_tokens must be >= 1.")

        self.channels = channels
        self.modulation_mode = modulation_mode
        self.num_text_tokens = num_text_tokens
        hidden = max(channels // reduction, 16)
        self.text_to_film = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        self.edge_to_q = nn.Linear(channels, channels)
        self.text_to_tokens = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Linear(hidden, num_text_tokens * channels),
        )
        self.text_to_k = nn.Linear(channels, channels)
        self.text_to_v = nn.Linear(channels, channels)
        self.attn_out = nn.Linear(channels, channels)
        self.attn_norm = nn.LayerNorm(channels)
        self.post = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    def _film_modulation(self, boundary_feature: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        text_embedding = _safe_repeat_text(text_embedding, boundary_feature.size(0))
        gamma_beta = self.text_to_film(text_embedding)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.sigmoid(gamma).view(boundary_feature.size(0), -1, 1, 1, 1)
        beta = beta.view(boundary_feature.size(0), -1, 1, 1, 1)
        return self.post(boundary_feature * gamma + beta)

    def _cross_attention_modulation(
        self,
        boundary_feature: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        text_embedding = _safe_repeat_text(text_embedding, boundary_feature.size(0))
        b, c, h, w, d = boundary_feature.shape

        # Edge tokens are Q: (B, H*W*D, C). This preserves anisotropic 3D
        # shapes because the original (H, W, D) is restored after attention.
        edge_tokens = boundary_feature.flatten(2).transpose(1, 2)
        q = self.edge_to_q(edge_tokens)

        # A single metadata vector is expanded into several text tokens. With
        # multiple K/V tokens, Q can select different semantic text components.
        text_tokens = self.text_to_tokens(text_embedding).view(b, self.num_text_tokens, c)
        k = self.text_to_k(text_tokens)
        v = self.text_to_v(text_tokens)

        attn_logits = torch.bmm(q, k.transpose(1, 2)) / (c ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)
        attended = torch.bmm(attn, v)
        attended = self.attn_norm(edge_tokens + self.attn_out(attended))
        attended = attended.transpose(1, 2).reshape(b, c, h, w, d)
        return self.post(attended)

    def forward(self, boundary_feature: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        if self.modulation_mode == "film":
            return self._film_modulation(boundary_feature, text_embedding)
        if self.modulation_mode == "cross_attention":
            return self._cross_attention_modulation(boundary_feature, text_embedding)
        raise RuntimeError(f"Unsupported modulation_mode: {self.modulation_mode}")


class AdaptiveBoundaryFusionGate3D(nn.Module):
    """
    Adaptively merge the image feature and text-guided boundary evidence.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2 + 1, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    def forward(
        self,
        feature: torch.Tensor,
        text_edge_feature: torch.Tensor,
        edge_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_prior = _resize_like(edge_prior, feature)
        gate = self.gate(torch.cat([feature, text_edge_feature, edge_prior], dim=1))
        fused_residual = self.out(gate * text_edge_feature)
        return feature + fused_residual, gate


class TextGuidedBoundaryFusion3D(nn.Module):
    """
    Full module: text-guided boundary structure attention fusion.

    Args:
        image_channels: number of channels in mix_image/input image.
        feature_channels: number of channels in the feature map to enhance.
        text_dim: dimension of metadata text embedding, default 512.
        band_width: boundary band width used for mix_mask-derived cut edges.

    Forward args:
        image: 5D mix image, (B, C_img, H, W, D).
        feature: 5D feature map to enhance, (B, C_feat, Hf, Wf, Df).
        text_embedding: metadata text embedding, (1, 512) or (B, 512).
        mix_mask: optional 5D mix mask, (B, 1, H, W, D). If unavailable, the
            module falls back to image high-frequency edges only.

    Returns:
        fused_feature: enhanced feature with the same shape as feature.
        aux: intermediate maps for visualization or auxiliary losses.
    """

    def __init__(
        self,
        image_channels: int,
        feature_channels: int,
        text_dim: int = DIM_TEXT_EMB,
        band_width: int = 3,
        operator_mode: str = DEFAULT_EDGE_OPERATOR_MODE,
        text_modulation_mode: str = DEFAULT_TEXT_MODULATION_MODE,
        num_text_tokens: int = 4,
    ):
        super().__init__()
        self.boundary_prior = BoundaryPriorFusion3D(
            image_channels=image_channels,
            band_width=band_width,
            operator_mode=operator_mode,
        )
        self.boundary_feature = BoundaryFeatureExtractor3D(feature_channels)
        self.text_modulator = TextBoundaryModulator3D(
            feature_channels,
            text_dim=text_dim,
            modulation_mode=text_modulation_mode,
            num_text_tokens=num_text_tokens,
        )
        self.fusion_gate = AdaptiveBoundaryFusionGate3D(feature_channels)

    def forward(
        self,
        image: torch.Tensor,
        feature: torch.Tensor,
        text_embedding: torch.Tensor,
        mix_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        edge_prior = self.boundary_prior(image=image, feature=feature, mix_mask=mix_mask)
        edge_feature = self.boundary_feature(feature=feature, edge_prior=edge_prior)
        text_edge_feature = self.text_modulator(edge_feature, text_embedding=text_embedding)
        fused_feature, fusion_gate = self.fusion_gate(
            feature=feature,
            text_edge_feature=text_edge_feature,
            edge_prior=edge_prior,
        )
        aux = {
            "edge_prior": edge_prior,
            "edge_feature": edge_feature,
            "text_edge_feature": text_edge_feature,
            "fusion_gate": fusion_gate,
        }
        return fused_feature, aux


class MultiScaleTextGuidedBoundaryFusion3D(nn.Module):
    """
    Optional wrapper for BiomedVNet-style encoder features.

    It applies TextGuidedBoundaryFusion3D to the last text_fuse_level feature
    maps in a feature list [x1, x2, x3, x4, x5]. This mirrors BiomedVNet's
    existing multi-scale text fusion policy.
    """

    def __init__(
        self,
        image_channels: int,
        feature_channels,
        text_dim: int = DIM_TEXT_EMB,
        text_fuse_level: int = 3,
        band_width: int = 3,
        operator_mode: str = DEFAULT_EDGE_OPERATOR_MODE,
        text_modulation_mode: str = DEFAULT_TEXT_MODULATION_MODE,
        num_text_tokens: int = 4,
    ):
        super().__init__()
        self.feature_channels = list(feature_channels)
        self.text_fuse_level = text_fuse_level
        fuse_start = len(self.feature_channels) - text_fuse_level
        self.blocks = nn.ModuleList()
        for idx, channels in enumerate(self.feature_channels):
            if idx >= fuse_start:
                self.blocks.append(
                    TextGuidedBoundaryFusion3D(
                        image_channels=image_channels,
                        feature_channels=channels,
                        text_dim=text_dim,
                        band_width=band_width,
                        operator_mode=operator_mode,
                        text_modulation_mode=text_modulation_mode,
                        num_text_tokens=num_text_tokens,
                    )
                )
            else:
                self.blocks.append(None)

    def forward(
        self,
        image: torch.Tensor,
        features,
        text_embedding: torch.Tensor,
        mix_mask: Optional[torch.Tensor] = None,
    ):
        fused_features = []
        aux_by_level = {}
        for idx, (feature, block) in enumerate(zip(features, self.blocks)):
            if block is None:
                fused_features.append(feature)
                continue
            fused, aux = block(
                image=image,
                feature=feature,
                text_embedding=text_embedding,
                mix_mask=mix_mask,
            )
            fused_features.append(fused)
            aux_by_level[f"level_{idx + 1}"] = aux
        return fused_features, aux_by_level


# =============================================================================
# Main: test data flow with random tensors
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # Common shapes
    B, C_img = 2, 1
    H, W, D = 64, 64, 48
    patch_size = (H, W, D)

    # Feature channels mirror BiomedVNet: [16, 32, 64, 128, 256]
    feature_channels = [16, 32, 64, 128, 256]
    text_dim = 512
    text_fuse_level = 3

    print("=" * 70)
    print("Testing edge_enhance_and_fusion.py modules")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Test CutBoundaryExtractor3D
    # -------------------------------------------------------------------------
    print("\n[1] CutBoundaryExtractor3D")
    mix_mask = torch.randint(0, 2, (B, 1, H, W, D)).float()
    ref = torch.randn(B, 16, H, W, D)
    extractor = CutBoundaryExtractor3D(band_width=3)
    cut_edge = extractor(mix_mask, ref)
    print(f"    mix_mask:    {tuple(mix_mask.shape)}")
    print(f"    ref:         {tuple(ref.shape)}")
    print(f"    cut_edge:    {tuple(cut_edge.shape)}  (boundary band from mix_mask)")

    # -------------------------------------------------------------------------
    # 2. Test ImageEdgeExtractor3D
    # -------------------------------------------------------------------------
    print("\n[2] ImageEdgeExtractor3D")
    image = torch.randn(B, C_img, H, W, D)
    img_edge_extractor = ImageEdgeExtractor3D(in_channels=C_img)
    image_edge = img_edge_extractor(image, ref)
    print(f"    image:       {tuple(image.shape)}")
    print(f"    ref:         {tuple(ref.shape)}")
    print(f"    image_edge:  {tuple(image_edge.shape)}  (Sobel gradient magnitude)")

    # -------------------------------------------------------------------------
    # 3. Test BoundaryPriorFusion3D
    # -------------------------------------------------------------------------
    print("\n[3] BoundaryPriorFusion3D")
    boundary_prior_fusion = BoundaryPriorFusion3D(image_channels=C_img, band_width=3)
    edge_prior = boundary_prior_fusion(image=image, feature=ref, mix_mask=mix_mask)
    print(f"    image:       {tuple(image.shape)}")
    print(f"    feature:     {tuple(ref.shape)}")
    print(f"    mix_mask:    {tuple(mix_mask.shape)}")
    print(f"    edge_prior:  {tuple(edge_prior.shape)}  (soft attention in [0,1])")

    # -------------------------------------------------------------------------
    # 4. Test BoundaryFeatureExtractor3D
    # -------------------------------------------------------------------------
    print("\n[4] BoundaryFeatureExtractor3D")
    C_feat = 64
    boundary_feature_extractor = BoundaryFeatureExtractor3D(channels=C_feat)
    feature_64 = torch.randn(B, C_feat, H // 4, W // 4, D // 4)
    boundary_feature = boundary_feature_extractor(feature_64, edge_prior)
    print(f"    feature:        {tuple(feature_64.shape)}")
    print(f"    edge_prior:     {tuple(edge_prior.shape)}  (will be resized)")
    print(f"    boundary_feat:  {tuple(boundary_feature.shape)}  (feature * edge_prior)")

    # -------------------------------------------------------------------------
    # 5. Test TextBoundaryModulator3D (FiLM)
    # -------------------------------------------------------------------------
    print("\n[5] TextBoundaryModulator3D (FiLM)")
    text_modulator = TextBoundaryModulator3D(channels=C_feat, text_dim=text_dim)
    text_embedding = torch.randn(B, text_dim)
    text_edge_feature = text_modulator(boundary_feature, text_embedding)
    print(f"    boundary_feature:  {tuple(boundary_feature.shape)}")
    print(f"    text_embedding:   {tuple(text_embedding.shape)}")
    print(f"    text_edge_feature: {tuple(text_edge_feature.shape)}")
    print(f"    FiLM gamma range: [{text_modulator.text_to_film(text_embedding)[:, :C_feat].min():.3f}, "
          f"{text_modulator.text_to_film(text_embedding)[:, :C_feat].max():.3f}]")

    # -------------------------------------------------------------------------
    # 6. Test AdaptiveBoundaryFusionGate3D
    # -------------------------------------------------------------------------
    print("\n[6] AdaptiveBoundaryFusionGate3D")
    fusion_gate = AdaptiveBoundaryFusionGate3D(channels=C_feat)
    fused_feature, gate = fusion_gate(
        feature=feature_64,
        text_edge_feature=text_edge_feature,
        edge_prior=edge_prior,
    )
    print(f"    feature:          {tuple(feature_64.shape)}")
    print(f"    text_edge_feat:   {tuple(text_edge_feature.shape)}")
    print(f"    edge_prior:       {tuple(edge_prior.shape)}")
    print(f"    fused_feature:    {tuple(fused_feature.shape)}  (residual added)")
    print(f"    gate:             {tuple(gate.shape)}  (in [0,1])")

    # -------------------------------------------------------------------------
    # 7. Test TextGuidedBoundaryFusion3D (full single-scale)
    # -------------------------------------------------------------------------
    print("\n[7] TextGuidedBoundaryFusion3D (full single-scale)")
    full_fusion = TextGuidedBoundaryFusion3D(
        image_channels=C_img,
        feature_channels=C_feat,
        text_dim=text_dim,
        band_width=3,
    )
    text_embed = torch.randn(B, text_dim)
    fused_out, aux = full_fusion(
        image=image,
        feature=feature_64,
        text_embedding=text_embed,
        mix_mask=mix_mask,
    )
    print(f"    image:            {tuple(image.shape)}")
    print(f"    feature:          {tuple(feature_64.shape)}")
    print(f"    text_embedding:   {tuple(text_embed.shape)}")
    print(f"    mix_mask:         {tuple(mix_mask.shape)}")
    print(f"    fused_feature:    {tuple(fused_out.shape)}")
    print(f"    aux keys:         {list(aux.keys())}")

    # -------------------------------------------------------------------------
    # 8. Test MultiScaleTextGuidedBoundaryFusion3D (full multi-scale)
    # -------------------------------------------------------------------------
    print("\n[8] MultiScaleTextGuidedBoundaryFusion3D (multi-scale)")
    multi_scale_fusion = MultiScaleTextGuidedBoundaryFusion3D(
        image_channels=C_img,
        feature_channels=feature_channels,
        text_dim=text_dim,
        text_fuse_level=text_fuse_level,
        band_width=3,
        operator_mode=SUPPORTED_EDGE_OPERATOR_MODES[0],
        text_modulation_mode=SUPPORTED_TEXT_MODULATION_MODES[1],
    )

    # Build dummy feature pyramid (spatial sizes halving each level)
    features = [
        torch.randn(B, feature_channels[0], H, W, D),         # x1
        torch.randn(B, feature_channels[1], H // 2, W // 2, D // 2),  # x2
        torch.randn(B, feature_channels[2], H // 4, W // 4, D // 4),  # x3
        torch.randn(B, feature_channels[3], H // 8, W // 8, D // 8),  # x4
        torch.randn(B, feature_channels[4], H // 16, W // 16, D // 16),  # x5
    ]
    print(f"    Input features:  {[tuple(f.shape) for f in features]}")

    fused_features, aux_by_level = multi_scale_fusion(
        image=image,
        features=features,
        text_embedding=text_embed,
        mix_mask=mix_mask,
    )
    print(f"    Output features:  {[tuple(f.shape) for f in fused_features]}")
    print(f"    Fused levels:    {[i for i, f in enumerate(fused_features) if f.shape != features[i].shape]}")
    print(f"    Aux level keys:   {list(aux_by_level.keys())}")
    for level_key, level_aux in aux_by_level.items():
        print(f"      {level_key} aux keys: {list(level_aux.keys())}")

    # -------------------------------------------------------------------------
    # 10. Verify shape consistency
    # -------------------------------------------------------------------------
    print("\n[10] Shape consistency check")
    all_ok = True
    for i, (orig, fused) in enumerate(zip(features, fused_features)):
        if orig.shape == fused.shape:
            print(f"    level_{i + 1}: {tuple(orig.shape)} -> (unchanged, no fusion block)")
        else:
            print(f"    level_{i + 1}: {tuple(orig.shape)} -> {tuple(fused.shape)} -> (fused)")
    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)
