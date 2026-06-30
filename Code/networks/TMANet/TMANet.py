from typing import Optional, Tuple

import torch

from Code.networks.TMANet.TMABaseNet import (
    TMA_Base_Net,
    DIM_TEXT_EMB,
)
from Code.networks.TMANet.TESF import (
    DEFAULT_EDGE_OPERATOR_MODE,
    DEFAULT_TEXT_MODULATION_MODE,
    MultiScaleTextGuidedBoundaryFusion3D,
)


def _get_mix_mask_from_metadata(metadata, explicit_mix_mask=None):
    """
    Resolve the mix/cut mask used by the boundary fusion module.

    Priority:
        1. explicit mix_mask argument from forward()
        2. metadata["mix_mask"]
        3. metadata["bcp_mask"]
        4. metadata["mask"]

    If no mask is available, the edge module will fall back to image-derived
    high-frequency edges only.
    """
    if explicit_mix_mask is not None:
        return explicit_mix_mask
    if metadata is None:
        return None
    for key in ("mix_mask", "bcp_mask", "mask"):
        if key in metadata:
            return metadata[key]
    return None


class TMA_Net(TMA_Base_Net):
    """
    TMA_Base_Net with Text-guided Mix-boundary Attention fusion.

    Pipeline:
        1. TMA_Base_Net encoder extracts multi-scale 3D image features.
        2. Existing text-to-vision projection fuses metadata text embedding
           into selected encoder scales.
        3. TMA fusion explicitly focuses on mix_image cut edges:
              cut/image edge prior
           -> edge feature extraction
           -> text boundary modulation
           -> adaptive boundary fusion gate
        4. TMA_Base_Net decoder predicts the final segmentation.

    All tensors are 5D feature volumes: (B, C, H, W, D). The TMA module uses
    interpolation by explicit target size, so H, W, and D do not need to be
    equal.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        n_channels=1,
        n_classes=2,
        n_filters=16,
        normalization="instancenorm",
        has_dropout=False,
        text_fuse_level=3,
        tma_fuse_level=3,
        edge_operator_mode: str = DEFAULT_EDGE_OPERATOR_MODE,
        text_modulation_mode: str = DEFAULT_TEXT_MODULATION_MODE,
        boundary_band_width: int = 3,
        num_text_tokens: int = 4,
    ):
        super(TMA_Net, self).__init__(
            patch_size=patch_size,
            n_channels=n_channels,
            n_classes=n_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            text_fuse_level=text_fuse_level,
        )

        self.tma_fuse_level = tma_fuse_level
        self.edge_operator_mode = edge_operator_mode
        self.text_modulation_mode = text_modulation_mode

        self.tma_fusion_layer = MultiScaleTextGuidedBoundaryFusion3D(
            image_channels=n_channels,
            feature_channels=self.feature_channels,
            text_dim=DIM_TEXT_EMB,
            text_fuse_level=tma_fuse_level,
            band_width=boundary_band_width,
            operator_mode=edge_operator_mode,
            text_modulation_mode=text_modulation_mode,
            num_text_tokens=num_text_tokens,
        )

    def forward(
        self,
        input,
        metadata,
        bcp=True,
        bcp_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        turnoff_drop: bool = True,
    ):
        """
        Args:
            input: mix_image or normal image, shape (B, C, H, W, D).
            metadata: dictionary containing at least "bcp" and "nobcp" text
                embeddings of shape (1, 512) or (B, 512). It may also contain
                "mix_mask", "bcp_mask", or "mask" for the cut boundary prior.
            bcp: choose metadata["bcp"] if True, otherwise metadata["nobcp"].
            bcp_mask: optional explicit cut mask. This overrides metadata masks.
            return_aux: return TMA intermediate maps when True.
            turnoff_drop: temporarily disable dropout, matching TMA_Base_Net.

        Returns:
            out: segmentation logits.
            aux or None: per-level TMA maps when return_aux=True.
        """
        text_embed = metadata["bcp"] if bcp else metadata["nobcp"]
        if text_embed.size(0) == 1:
            text_embed = text_embed.repeat(input.size(0), 1)

        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        # 1. Standard TMA_Base_Net encoder.
        features = self.encoder(input)


        # 2. New TMA fusion focused on mix_image cut edges and boundary context.
        resolved_mix_mask = _get_mix_mask_from_metadata(metadata, explicit_mix_mask=bcp_mask)
        if resolved_mix_mask!=None and len(resolved_mix_mask.size())!= 5:
            resolved_mix_mask = resolved_mix_mask.unsqueeze(dim=1)
        tma_features, tma_aux = self.tma_fusion_layer(
            image=input,
            features=features,
            text_embedding=text_embed,
            mix_mask=resolved_mix_mask,
        )

        # 4. Standard TMA_Base_Net decoder.
        out = self.decoder(tma_features)

        if turnoff_drop:
            self.has_dropout = has_dropout

        return out, tma_aux if return_aux else None

