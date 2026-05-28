"""
models/unet.py — U-Net base para estudio de ablación.

Arquitectura estándar sin Attention Gates.
Usada en el estudio de ablación para comparar con Attention U-Net.

Referencia:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation," MICCAI 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Bloque doble convolucional: (Conv → BN → ReLU) × 2"""
    def __init__(self, in_channels: int, out_channels: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net base sin Attention Gates.

    Parámetros
    ----------
    in_channels  : canales de entrada
    out_channels : canales de salida (1 para segmentación binaria)
    base_channels: filtros en primer bloque
    depth        : niveles del encoder/decoder
    bilinear     : upsampling bilineal o transpuesta
    dropout_rate : dropout en bloques convolucionales
    """

    def __init__(
        self,
        in_channels:   int   = 3,
        out_channels:  int   = 1,
        base_channels: int   = 64,
        depth:         int   = 4,
        bilinear:      bool  = True,
        dropout_rate:  float = 0.1,
    ):
        super().__init__()
        self.depth = depth
        c = base_channels

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = c * (2 ** i)
            self.encoders.append(
                ConvBlock(in_ch, out_ch, dropout_rate)
            )
            self.pools.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(
            in_ch, c * (2 ** depth), dropout_rate
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.ups   = nn.ModuleList()
        self.decs  = nn.ModuleList()
        in_ch = c * (2 ** depth)
        for i in range(depth - 1, -1, -1):
            skip_ch = c * (2 ** i)
            out_ch  = skip_ch
            if bilinear:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2,
                                mode="bilinear", align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                ))
            else:
                self.ups.append(
                    nn.ConvTranspose2d(in_ch, out_ch,
                                      kernel_size=2, stride=2)
                )
            self.decs.append(
                ConvBlock(out_ch + skip_ch, out_ch, dropout_rate)
            )
            in_ch = out_ch

        self.output_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.output_conv(x)

    def predict_mask(self, x: torch.Tensor,
                     threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return (torch.sigmoid(logits) > threshold).float()
