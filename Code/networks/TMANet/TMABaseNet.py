from typing import Tuple

import torch
from torch import nn

DIM_TEXT_EMB = 512

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:

            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

# ========================
# StackedFusionConvLayers
# ========================

class ConvDropoutNormNonlin(nn.Module):
    """
    Basic building block: Conv -> Dropout -> InstanceNorm -> LeakyReLU
    """
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)

        if dropout_op is not None and dropout_op_kwargs.get('p', 0) > 0:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None

        self.instnorm = norm_op(output_channels, **norm_op_kwargs)
        self.lrelu = nonlin(**nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.instnorm(x)
        x = self.lrelu(x)
        return x

class StackedFusionConvLayers(nn.Module):
    """
    Stacked Fusion Convolutional Layers for prompt-feature fusion.

    Architecture:
        Input: (B, C_in, H, W, D) - concatenated image features + prompt embeddings
        → 3 stacked ConvDropoutNormNonlin blocks
        → Output: (B, C_out, H, W, D)

    The fusion is performed at the bottleneck layer (after all pooling operations).
    This allows the text prompts to interact deeply with the image features.
    """
    def __init__(self, input_feature_channels, bottleneck_feature_channel, output_feature_channels, num_convs=3,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None):
        super(StackedFusionConvLayers, self).__init__()

        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        # Build stacked blocks
        blocks = []

        # First block: input -> bottleneck
        blocks.append(ConvDropoutNormNonlin(
            input_feature_channels, bottleneck_feature_channel,
            conv_op, conv_kwargs, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
        ))

        # Middle blocks: bottleneck -> bottleneck
        for _ in range(num_convs - 2):
            blocks.append(ConvDropoutNormNonlin(
                bottleneck_feature_channel, bottleneck_feature_channel,
                conv_op, conv_kwargs, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
            ))

        # Last block: bottleneck -> output
        blocks.append(ConvDropoutNormNonlin(
            bottleneck_feature_channel, output_feature_channels,
            conv_op, conv_kwargs, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
        ))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class Texts_To_Vision_Mutilscale(nn.Module):
    """
    Transform text embeddings (from CLIP) to match multi-scale feature map resolutions.
    For each encoder scale i, projects text_embed (B, 512) -> (B, 1, H_i, W_i, D_i)
    where (H_i, W_i, D_i) = patch_size / (2^i)
    Text embedding channel is always 1.
    Only projects for the last text_fuse_level scales.
    """
    def __init__(self, patch_size, feature_channels, text_embed_dim=DIM_TEXT_EMB, text_fuse_level=5):
        super(Texts_To_Vision_Mutilscale, self).__init__()

        self.patch_size = patch_size
        self.feature_channels = feature_channels  # list of 5 channel sizes
        self.text_ch = 1  # text embedding channel is always 1
        self.text_fuse_level = text_fuse_level

        # Linear projection only for last text_fuse_level scales
        self.projs = nn.ModuleList()
        for i in range(len(feature_channels)):
            if i >= len(feature_channels) - text_fuse_level:
                scale_dim = patch_size[0] * patch_size[1] * patch_size[2] // (2 ** (i * 3))
                self.projs.append(nn.Linear(text_embed_dim, scale_dim * self.text_ch))
            else:
                self.projs.append(None)

    def forward(self, text_embed):
        """
        Args:
            text_embed: (B, 512) - raw text embedding from CLIP
        Returns:
            list of 5 tensors, each (B, 1, H_i, W_i, D_i) matching corresponding feature resolution
            Non-fusion scales return None.
        """
        out = []
        for i, (proj, feat_ch) in enumerate(zip(self.projs, self.feature_channels)):
            H = self.patch_size[0] // (2 ** i)
            W = self.patch_size[1] // (2 ** i)
            D = self.patch_size[2] // (2 ** i)

            if proj is None:
                out.append(None)
            else:
                # Project: (B, 512) -> (B, scale_dim * 1)
                text_feat = proj(text_embed)
                # Reshape: (B, scale_dim * 1) -> (B, 1, H, W, D)
                text_feat = text_feat.reshape(-1, self.text_ch, H, W, D)
                out.append(text_feat)
        return out

class FusionModule(nn.Module):
    """
    Fuse multi-scale image features with text prompt embeddings at each scale.
    Takes concatenated [feature, text_prompt] and outputs fused feature.
    Only fuses for the last text_fuse_level scales.
    """
    def __init__(self, feature_channels, text_channels, fusion_bottleneck_ch=32, text_fuse_level=5):
        super(FusionModule, self).__init__()

        self.fusion_blocks = nn.ModuleList()
        self.text_fuse_level = text_fuse_level
        for idx, (feat_ch, txt_ch) in enumerate(zip(feature_channels, text_channels)):
            if idx >= len(feature_channels) - text_fuse_level:
                in_ch = feat_ch + txt_ch
                self.fusion_blocks.append(
                    StackedFusionConvLayers(
                        input_feature_channels=in_ch,
                        bottleneck_feature_channel=fusion_bottleneck_ch,
                        output_feature_channels=feat_ch,
                        num_convs=3,
                    )
                )
            else:
                self.fusion_blocks.append(None)

    def forward(self, features, text_embeds):
        """
        Args:
            features: list of 5 tensors (B, C_i, H_i, W_i, D_i) at scales 1, 1/2, 1/4, 1/8, 1/16
            text_embeds: list of 5 tensors (B, 1, H_i, W_i, D_i) matching feature resolutions, None for non-fusion scales
        Returns:
            list of 5 fused tensors (B, C_i, H_i, W_i, D_i)
        """
        fused = []
        for feat, text_emb, block in zip(features, text_embeds, self.fusion_blocks):
            if block is None:
                fused.append(feat)
            else:
                # Concatenate along channel dimension: (B, C_feat + C_text, H, W, D)
                fused_input = torch.cat([feat, text_emb], dim=1)
                fused.append(block(fused_input))
        return fused

class FusionModuleMLPConcatConv(nn.Module):
    """
    Ablation4-1:
    Remove stacked fusion blocks, keep:
    text MLP projection -> concat(feature, text) -> one conv restore channels.
    """
    def __init__(self, feature_channels, text_channels, text_fuse_level=5):
        super(FusionModuleMLPConcatConv, self).__init__()
        self.text_fuse_level = text_fuse_level
        self.restore_convs = nn.ModuleList()
        for idx, (feat_ch, txt_ch) in enumerate(zip(feature_channels, text_channels)):
            if idx >= len(feature_channels) - text_fuse_level:
                self.restore_convs.append(
                    nn.Conv3d(feat_ch + txt_ch, feat_ch, kernel_size=3, padding=1)
                )
            else:
                self.restore_convs.append(None)

    def forward(self, features, text_embeds):
        fused = []
        for feat, text_emb, conv in zip(features, text_embeds, self.restore_convs):
            if conv is None:
                fused.append(feat)
            else:
                fused_input = torch.cat([feat, text_emb], dim=1)
                fused.append(conv(fused_input))
        return fused

class FusionModuleChannelAdd(nn.Module):
    """
    Ablation4-2:
    Directly add projected text map to feature channels by broadcasting.
    """
    def __init__(self, feature_channels, text_fuse_level=5):
        super(FusionModuleChannelAdd, self).__init__()
        self.feature_channels = feature_channels
        self.text_fuse_level = text_fuse_level

    def forward(self, features, text_embeds):
        fused = []
        fuse_start = len(self.feature_channels) - self.text_fuse_level
        for idx, (feat, text_emb) in enumerate(zip(features, text_embeds)):
            if idx < fuse_start or text_emb is None:
                fused.append(feat)
            else:
                fused.append(feat + text_emb)
        return fused

class TMA_Base_Net(nn.Module):
    def __init__(self, patch_size:Tuple[int,int,int],
                 n_channels=1,
                 n_classes=2,
                 n_filters=16,
                 normalization='instancenorm',
                 has_dropout=False,
                 text_fuse_level=3):
        super(TMA_Base_Net, self).__init__()

        # Biomed
        self.patch_size = patch_size
        self.n_filters = n_filters
        self.text_fuse_level = text_fuse_level

        # Feature channels at each encoder scale
        self.feature_channels = [
            n_filters,        # x1: full res
            n_filters * 2,    # x2: 1/2
            n_filters * 4,    # x3: 1/4
            n_filters * 8,    # x4: 1/8
            n_filters * 16,   # x5: 1/16
        ]

        # Text-to-vision multi-scale projection (only last text_fuse_level scales)
        self.text_to_vision_ms = Texts_To_Vision_Mutilscale(
            patch_size=patch_size,
            feature_channels=self.feature_channels,
            text_embed_dim=DIM_TEXT_EMB,
            text_fuse_level=text_fuse_level,
        )

        # Multi-scale fusion layer (only last text_fuse_level scales)
        self.fusion_layer = FusionModule(
            feature_channels=self.feature_channels,
            text_channels=[1, 1, 1, 1, 1],  # text projected to single channel
            fusion_bottleneck_ch=32,
            text_fuse_level=text_fuse_level,
        )

        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)
        if has_dropout:
            self.dropout = nn.Dropout3d(p=0.5)
        self.branchs = nn.ModuleList()
        for i in range(1):
            if has_dropout:
                seq = nn.Sequential(
                    ConvBlock(1, n_filters, n_filters, normalization=normalization),
                    nn.Dropout3d(p=0.5),
                    nn.Conv3d(n_filters, n_classes, 1, padding=0)
                )
            else:
                seq = nn.Sequential(
                    ConvBlock(1, n_filters, n_filters, normalization=normalization),
                    nn.Conv3d(n_filters, n_classes, 1, padding=0)
                )
            self.branchs.append(seq)

    def encoder(self, input):

        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        for branch in self.branchs:
            o = branch(x8_up)
            out = o
        # out.append(x6)
        return out

    def forward(self, input, metadata, bcp=True, turnoff_drop=False):
        text_embed = metadata["bcp"] if bcp else metadata["nobcp"]
        text_embed = text_embed.repeat(input.size(0),1)
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        # Get multi-scale features from encoder: [x1, x2, x3, x4, x5]
        features = self.encoder(input)

        # Project text embedding to multi-scale resolution
        text_embeds = self.text_to_vision_ms(text_embed) # list of 5

        # Fuse features with text prompts at each scale
        fused_features = self.fusion_layer(features, text_embeds)  # list of 5

        out = self.decoder(fused_features)
        if turnoff_drop:
            self.has_dropout = has_dropout

        return out, None



if __name__ == '__main__':
    patch_size = (112, 112, 80)
    input_tensor = torch.randn((1, 1, *patch_size)).cuda()
    model = TMA_Base_Net(patch_size=patch_size, n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=False).cuda()
    metadata = {
        "bcp": torch.randn(1, 512).cuda(),
        "nobcp": torch.randn(1, 512).cuda(),
    }
    from torch.nn.functional import normalize
    for k in metadata:
        metadata[k] = normalize(metadata[k], dim=1)
    out, _ = model(input_tensor,metadata)
    print(f"Output shape: {out.shape}")
