"""
UVT2T-CNet Model Architecture
=============================
U-Net with Convolutional Block Attention Module (CBAM) for GOFLOW velocity prediction.

This architecture integrates CBAM attention after the first encoder block to enable
state-dependent feature refinement. The attention mechanism allows the network to
focus on spatially and channel-wise relevant features based on input content.

Key Design:
- CBAM applies sequential channel attention then spatial attention
- Final skip connection uses PRE-attention features (preserving fine details)
- Encoder path uses POST-attention features (refined context)

References:
    Woo, S., Park, J., Lee, J.Y., & Kweon, I.S. (2018). CBAM: Convolutional block
    attention module. In ECCV (pp. 3-19). https://doi.org/10.1007/978-3-030-01234-2_1

    Chen, X., Zheng, F., Xia, J., Zhu, J., Shu, Y., & Liu, D. (2025). High-resolution
    regional SST AI downscaling based on multi-mode inputs from nested ROMS simulations.
    Machine Learning: Science and Technology.

Author: Kaushik Srinivasan (UCLA Atmospheric and Oceanic Sciences)
"""

import torch
import torch.nn as nn
from unet_parts_t import Down, DoubleConv, Up


class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module.
    Aggregates spatial info using Avg and Max pooling, then processes via shared MLP.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention Module.
    Aggregates channel info via pooling, then applies 7x7 convolution.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise Avg and Max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequential application of Channel Attention then Spatial Attention.
    """
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel attention refinement
        x = x * self.ca(x)
        # Spatial attention refinement
        x = x * self.sa(x)
        return x

class UVT2T_CNet(nn.Module):
    """
    U-Net with CBAM attention for image-to-image prediction.

    Drop-in replacement for UNet implementing the UVT2T-CNet architecture.
    Includes CBAM after the first encoder block for state-dependent feature
    refinement. The skip connection to the decoder uses pre-attention features,
    while the encoder path continues with attention-refined features.

    Architecture:
        - Initial conv block → CBAM attention
        - 4 encoder blocks with strided convolutions (downsampling)
        - Bottleneck layer
        - 4 decoder blocks with skip connections (upsampling)
        - 1x1 output convolution

    Args:
        n_channels: Number of input channels (e.g., 3 for 3-frame SST input)
        no: Number of output channels (e.g., 2 for U,V velocity)
        bilinear: If True, use bilinear upsampling; if False, use transposed conv
        Nbase: Base channel count (model size scales with this)
        inpNorm: If True, apply batch normalization to input
    """

    def __init__(self, n_channels, no, bilinear=False, Nbase=16, inpNorm=False):
        super(UVT2T_CNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.inpNorm = inpNorm
        
        if inpNorm:
            self.bn = nn.BatchNorm2d(n_channels)
            
        # 1. Initial Double Conv (matches paper's 'Double Conv')
        self.inc = DoubleConv(n_channels, Nbase, stride=1)
        
        # 2. CBAM Module (The key addition)
        # ratio=16 is standard for CBAM; Nbase is usually small (16/32), ensure ratio doesn't reduce to 0
        reduction_ratio = 16 if Nbase >= 16 else 1 
        self.cbam = CBAM(Nbase, ratio=reduction_ratio)
        
        # 3. Encoder (Downsampling)
        # Note: Input to down1 is Nbase, output Nbase*2
        self.down1 = Down(Nbase, Nbase*2)
        self.down2 = Down(Nbase*2, Nbase*4)
        self.down3 = Down(Nbase*4, Nbase*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(Nbase*8, Nbase*16 // factor)
        
        # 4. Decoder (Upsampling)
        self.up1 = Up(Nbase*16, Nbase*8 // factor, bilinear)
        self.up2 = Up(Nbase*8, Nbase*4 // factor, bilinear)
        self.up3 = Up(Nbase*4, Nbase*2 // factor, bilinear)
        self.up4 = Up(Nbase*2, Nbase, bilinear)
        
        # 5. Output
        self.outc = nn.Conv2d(Nbase, no, kernel_size=1)

    def forward(self, x):
        # Input Normalization
        if self.inpNorm:
            x = self.bn(x)
            
        # --- Encoder ---
        x1 = self.inc(x)  # Original features (Pre-Attention)
        
        # Apply CBAM
        x1_att = self.cbam(x1) # Refined features (Post-Attention)
        
        # Downsampling path uses Attended features
        x2 = self.down1(x1_att)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # --- Decoder ---
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        
        # Final Up: 
        # Paper 2.2.2: "first skip connection retains the original (pre-attention) features"
        # So we skip-connect x1 (pre-CBAM), not x1_att
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.cbam = torch.utils.checkpoint(self.cbam) # Checkpoint CBAM too
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
