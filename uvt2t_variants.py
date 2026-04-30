"""Two CBAM-placement variants of UVT2T-CNet for attention-placement experiments.

- UVT2T_Multi: CBAM after every encoder level (5 modules total).
- UVT2T_Bot:   CBAM only at the bottleneck (1 module).

Skip-connection convention matches the original UVT2T (Woo et al., 2018; Chen et al., 2025):
encoder downsamples use attention-refined features; skip connections to the
decoder use PRE-attention features to preserve fine detail.
"""
import torch
import torch.nn as nn
from models.unet_parts_t import DoubleConv, Down, Up
from models.uvt2t_cnet import CBAM


class UVT2T_Multi(nn.Module):
    """UNet with CBAM at every encoder level (inc, down1, down2, down3, down4)."""

    def __init__(self, n_channels, no, bilinear=False, Nbase=16, inpNorm=False):
        super().__init__()
        self.bilinear = bilinear
        self.inpNorm = inpNorm
        if inpNorm:
            self.bn = nn.BatchNorm2d(n_channels)

        self.inc = DoubleConv(n_channels, Nbase, stride=1)
        r = 16 if Nbase >= 16 else 1
        self.cbam1 = CBAM(Nbase,     ratio=r)
        self.cbam2 = CBAM(Nbase * 2, ratio=r)
        self.cbam3 = CBAM(Nbase * 4, ratio=r)
        self.cbam4 = CBAM(Nbase * 8, ratio=r)

        factor = 2 if bilinear else 1
        self.down1 = Down(Nbase,     Nbase * 2)
        self.down2 = Down(Nbase * 2, Nbase * 4)
        self.down3 = Down(Nbase * 4, Nbase * 8)
        self.down4 = Down(Nbase * 8, Nbase * 16 // factor)
        self.cbam5 = CBAM(Nbase * 16 // factor, ratio=r)

        self.up1 = Up(Nbase * 16, Nbase * 8 // factor, bilinear)
        self.up2 = Up(Nbase * 8,  Nbase * 4 // factor, bilinear)
        self.up3 = Up(Nbase * 4,  Nbase * 2 // factor, bilinear)
        self.up4 = Up(Nbase * 2,  Nbase,               bilinear)
        self.outc = nn.Conv2d(Nbase, no, kernel_size=1)

    def forward(self, x):
        if self.inpNorm:
            x = self.bn(x)

        # Encoder: compute pre-attention features, attention-refined features flow downward,
        # pre-attention features are held for skip connections.
        x1 = self.inc(x)
        x1_a = self.cbam1(x1)

        x2 = self.down1(x1_a)
        x2_a = self.cbam2(x2)

        x3 = self.down2(x2_a)
        x3_a = self.cbam3(x3)

        x4 = self.down3(x3_a)
        x4_a = self.cbam4(x4)

        x5 = self.down4(x4_a)
        x5_a = self.cbam5(x5)

        # Decoder: skips use pre-attention features (x1, x2, x3, x4).
        x = self.up1(x5_a, x4)
        x = self.up2(x,    x3)
        x = self.up3(x,    x2)
        x = self.up4(x,    x1)
        return self.outc(x)


class UVT2T_FullAttn(nn.Module):
    """UNet with CBAM at every encoder level AND after every decoder Up block (post-skip).
    Encoder: same as UVT2T_Multi (CBAM after inc, down1, down2, down3, down4).
    Decoder: CBAM after each Up's double-conv, on the concat'd skip-merged features.
    Total: 9 CBAMs (5 encoder + 4 decoder). Skip connections still use pre-attention features
    from the encoder side."""

    def __init__(self, n_channels, no, bilinear=False, Nbase=16, inpNorm=False):
        super().__init__()
        self.bilinear = bilinear
        self.inpNorm = inpNorm
        if inpNorm:
            self.bn = nn.BatchNorm2d(n_channels)

        r = 16 if Nbase >= 16 else 1

        self.inc = DoubleConv(n_channels, Nbase, stride=1)
        self.cbam1 = CBAM(Nbase,     ratio=r)
        self.cbam2 = CBAM(Nbase * 2, ratio=r)
        self.cbam3 = CBAM(Nbase * 4, ratio=r)
        self.cbam4 = CBAM(Nbase * 8, ratio=r)

        factor = 2 if bilinear else 1
        self.down1 = Down(Nbase,     Nbase * 2)
        self.down2 = Down(Nbase * 2, Nbase * 4)
        self.down3 = Down(Nbase * 4, Nbase * 8)
        self.down4 = Down(Nbase * 8, Nbase * 16 // factor)
        self.cbam5 = CBAM(Nbase * 16 // factor, ratio=r)

        self.up1 = Up(Nbase * 16, Nbase * 8 // factor, bilinear)
        self.up2 = Up(Nbase * 8,  Nbase * 4 // factor, bilinear)
        self.up3 = Up(Nbase * 4,  Nbase * 2 // factor, bilinear)
        self.up4 = Up(Nbase * 2,  Nbase,               bilinear)
        # Post-skip decoder CBAMs at each Up's output
        self.cbam_dec1 = CBAM(Nbase * 8 // factor, ratio=r)
        self.cbam_dec2 = CBAM(Nbase * 4 // factor, ratio=r)
        self.cbam_dec3 = CBAM(Nbase * 2 // factor, ratio=r)
        self.cbam_dec4 = CBAM(Nbase,               ratio=r)

        self.outc = nn.Conv2d(Nbase, no, kernel_size=1)

    def forward(self, x):
        if self.inpNorm:
            x = self.bn(x)

        x1 = self.inc(x)
        x1_a = self.cbam1(x1)
        x2 = self.down1(x1_a); x2_a = self.cbam2(x2)
        x3 = self.down2(x2_a); x3_a = self.cbam3(x3)
        x4 = self.down3(x3_a); x4_a = self.cbam4(x4)
        x5 = self.down4(x4_a); x5_a = self.cbam5(x5)

        # Decoder: skips use pre-attention encoder features (x1..x4),
        # CBAM applied AFTER each Up block on the merged features.
        x = self.cbam_dec1(self.up1(x5_a, x4))
        x = self.cbam_dec2(self.up2(x,    x3))
        x = self.cbam_dec3(self.up3(x,    x2))
        x = self.cbam_dec4(self.up4(x,    x1))
        return self.outc(x)


class UVT2T_Bot(nn.Module):
    """UNet with CBAM only at the bottleneck (between deepest encoder and first decoder up)."""

    def __init__(self, n_channels, no, bilinear=False, Nbase=16, inpNorm=False):
        super().__init__()
        self.bilinear = bilinear
        self.inpNorm = inpNorm
        if inpNorm:
            self.bn = nn.BatchNorm2d(n_channels)

        self.inc = DoubleConv(n_channels, Nbase, stride=1)
        factor = 2 if bilinear else 1
        self.down1 = Down(Nbase,     Nbase * 2)
        self.down2 = Down(Nbase * 2, Nbase * 4)
        self.down3 = Down(Nbase * 4, Nbase * 8)
        self.down4 = Down(Nbase * 8, Nbase * 16 // factor)

        r = 16 if Nbase >= 16 else 1
        self.cbam_bot = CBAM(Nbase * 16 // factor, ratio=r)

        self.up1 = Up(Nbase * 16, Nbase * 8 // factor, bilinear)
        self.up2 = Up(Nbase * 8,  Nbase * 4 // factor, bilinear)
        self.up3 = Up(Nbase * 4,  Nbase * 2 // factor, bilinear)
        self.up4 = Up(Nbase * 2,  Nbase,               bilinear)
        self.outc = nn.Conv2d(Nbase, no, kernel_size=1)

    def forward(self, x):
        if self.inpNorm:
            x = self.bn(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_a = self.cbam_bot(x5)
        x = self.up1(x5_a, x4)
        x = self.up2(x,    x3)
        x = self.up3(x,    x2)
        x = self.up4(x,    x1)
        return self.outc(x)
