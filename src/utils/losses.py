"""
utils/losses.py — Funciones de pérdida para segmentación de vasos retinianos.

Implementa:
  - BCEWithLogitsLoss : pérdida estándar para segmentación binaria
  - DiceLoss          : pérdida basada en coeficiente Dice
  - CombinedLoss      : BCE + Dice (la más efectiva para vasos finos)

Referencia Dice:
    Milletari et al., "V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation," 3DV 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy con logits.
    Soporta pesos por píxel para clases desbalanceadas
    (vasos ocupan ~10% de la imagen).
    """
    def __init__(self, pos_weight: float = None):
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(
        self,
        logits: torch.Tensor,   # (B, 1, H, W) sin sigmoid
        targets: torch.Tensor,  # (B, 1, H, W) binaria [0, 1]
    ) -> torch.Tensor:
        return self.criterion(logits, targets.float())


class DiceLoss(nn.Module):
    """
    Dice Loss para segmentación binaria.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice

    Ventaja sobre BCE: no se ve afectada por el desbalance de clases
    (vasos son ~10% de los píxeles, fondo ~90%).

    Parámetros
    ----------
    smooth : float  factor de suavizado para evitar división por cero
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        proba   = torch.sigmoid(logits)
        targets = targets.float()

        # Aplanar dimensiones espaciales
        proba   = proba.view(proba.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (proba * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / \
               (proba.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Pérdida combinada: α × BCE + (1-α) × Dice

    Esta combinación es la más efectiva para segmentación de vasos
    retinianos porque:
    - BCE asegura precisión píxel a píxel
    - Dice maneja el desbalance de clases (vasos vs fondo)

    Parámetros
    ----------
    alpha      : float  peso de BCE (default 0.5)
    pos_weight : float  peso positivo para BCE (default None)
    smooth     : float  suavizado para Dice
    """
    def __init__(
        self,
        alpha:      float = 0.5,
        pos_weight: float = None,
        smooth:     float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.bce   = BCELoss(pos_weight=pos_weight)
        self.dice  = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


def get_loss(loss_name: str, cfg: dict) -> nn.Module:
    """
    Factory de funciones de pérdida.

    Parámetros
    ----------
    loss_name : 'bce' | 'dice' | 'combined'
    cfg       : configuración del experimento
    """
    pos_weight = cfg.get("pos_weight", None)
    smooth     = cfg.get("dice_smooth", 1.0)
    alpha      = cfg.get("combined_alpha", 0.5)

    if loss_name == "bce":
        return BCELoss(pos_weight=pos_weight)
    elif loss_name == "dice":
        return DiceLoss(smooth=smooth)
    elif loss_name == "combined":
        return CombinedLoss(
            alpha=alpha,
            pos_weight=pos_weight,
            smooth=smooth,
        )
    else:
        raise ValueError(
            f"Pérdida desconocida: '{loss_name}'. "
            "Usa 'bce', 'dice' o 'combined'."
        )
