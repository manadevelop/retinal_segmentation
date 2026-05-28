"""
models/attention_unet.py — Attention U-Net para segmentación de vasos retinianos.

Arquitectura:
  - Encoder: 4 niveles con bloques dobles convolucionales + MaxPool
  - Bottleneck: bloque convolucional profundo
  - Decoder: 4 niveles con Attention Gates + UpConv + concatenación
  - Salida: mapa de segmentación binaria (vasos / fondo)

Referencia:
    Oktay et al., "Attention U-Net: Learning Where to Look for
    the Pancreas," MIDL 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Bloques fundamentales ──────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Bloque doble convolucional: (Conv → BN → ReLU) × 2
    Es el bloque básico del encoder y decoder de U-Net.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
    ):
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention Gate para Attention U-Net.

    Filtra las skip connections del encoder, enfocándose en las
    regiones relevantes (vasos) y suprimiendo el fondo.

    Parámetros
    ----------
    F_g : int   canales de la señal gating (del decoder)
    F_l : int   canales de la skip connection (del encoder)
    F_int : int canales intermedios del gate

    Referencia:
        Oktay et al., "Attention U-Net," MIDL 2018.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        # Transformación lineal de la señal gating (decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Transformación lineal de la skip connection (encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Capa de salida del attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        g: torch.Tensor,   # señal del decoder  (B, F_g, H, W)
        x: torch.Tensor,   # skip connection     (B, F_l, H, W)
    ) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Sumar señales (pueden tener diferente resolución espacial)
        if g1.shape != x1.shape:
            g1 = F.interpolate(
                g1, size=x1.shape[2:],
                mode="bilinear", align_corners=False
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)           # mapa de atención en [0, 1]
        return x * psi                # skip connection filtrada


class UpBlock(nn.Module):
    """
    Bloque de upsampling del decoder:
      1. UpConv (transpuesta o bilineal)
      2. Attention Gate sobre la skip connection
      3. Concatenación con skip connection filtrada
      4. ConvBlock doble
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2
            )

        self.att = AttentionGate(
            F_g=out_channels,
            F_l=skip_channels,
            F_int=out_channels // 2,
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x    = self.up(x)
        skip = self.att(g=x, x=skip)

        # Ajustar tamaño si hay diferencia por padding
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:],
                mode="bilinear", align_corners=False
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ── Attention U-Net completa ───────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net para segmentación binaria de vasos retinianos.

    Parámetros
    ----------
    in_channels  : int   canales de entrada (3 para RGB, 1 para gris)
    out_channels : int   canales de salida (1 para segmentación binaria)
    base_channels: int   filtros en el primer bloque (se duplican por nivel)
    depth        : int   número de niveles del encoder/decoder (3 o 4)
    bilinear     : bool  usar upsampling bilineal (True) o transpuesta (False)
    dropout_rate : float dropout en los bloques convolucionales
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
                ConvBlock(in_ch, out_ch, dropout_rate=dropout_rate)
            )
            self.pools.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch

        # ── Bottleneck ────────────────────────────────────────────────────
        bottleneck_ch = c * (2 ** depth)
        self.bottleneck = ConvBlock(
            in_ch, bottleneck_ch, dropout_rate=dropout_rate
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        in_ch = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = c * (2 ** i)
            out_ch  = skip_ch
            self.decoders.append(
                UpBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    bilinear=bilinear,
                    dropout_rate=dropout_rate,
                )
            )
            in_ch = out_ch

        # ── Salida ────────────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(
            in_ch, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : (B, C, H, W)

        Devuelve
        --------
        logits : (B, 1, H, W)  — sin sigmoid (para BCEWithLogitsLoss)
        """
        # Encoder + guardar skip connections
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder + attention gates
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.output_conv(x)

    def predict_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Devuelve máscara binaria predicha.

        Parámetros
        ----------
        x         : (B, C, H, W)
        threshold : umbral de binarización

        Devuelve
        --------
        mask : (B, 1, H, W) binaria
        """
        with torch.no_grad():
            logits = self.forward(x)
            proba  = torch.sigmoid(logits)
            return (proba > threshold).float()
