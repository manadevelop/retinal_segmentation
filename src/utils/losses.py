"""
Funciones de pérdida para segmentación binaria de vasos retinianos.

Incluye:
- BCEWithLogitsLoss
- DiceLoss
- BCE + Dice
- FocalLoss opcional

Todas las pérdidas son compatibles con CPU/GPU y evitan errores de device mismatch.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Binary Cross Entropy con logits.

    Soporta pos_weight para compensar desbalance entre fondo y vasos.
    El pos_weight se registra como buffer para que pueda moverse correctamente
    a CPU/GPU, y además se fuerza al device de logits en forward.
    """

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight",
                torch.tensor([float(pos_weight)], dtype=torch.float32)
            )
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        else:
            pos_weight = None

        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight
        )


class DiceLoss(nn.Module):
    """
    Dice Loss para segmentación binaria.

    Se aplica sigmoid a logits internamente.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Pérdida combinada BCE + Dice.

    Útil para segmentación retinal porque BCE estabiliza el aprendizaje píxel a píxel
    y Dice mejora el solapamiento de estructuras delgadas.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = BCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


class FocalLoss(nn.Module):
    """
    Focal Loss binaria con logits.

    Opcional para escenarios de fuerte desbalance foreground/background.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = (1 - pt).pow(self.gamma)
        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype),
            torch.tensor(1 - self.alpha, device=logits.device, dtype=logits.dtype),
        )

        loss = alpha_weight * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


def get_loss(
    name: str,
    pos_weight: Optional[float] = None,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    smooth: float = 1.0,
) -> nn.Module:
    """
    Factory de pérdidas.

    Nombres soportados:
    - bce
    - dice
    - bce_dice
    - combined
    - bce+dice
    - focal
    """

    name = name.lower().strip()

    if name in {"bce", "binary_cross_entropy"}:
        return BCELoss(pos_weight=pos_weight)

    if name in {"dice", "dice_loss"}:
        return DiceLoss(smooth=smooth)

    if name in {"bce_dice", "combined", "bce+dice", "dice_bce"}:
        return BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
            smooth=smooth,
        )

    if name in {"focal", "focal_loss"}:
        return FocalLoss()

    raise ValueError(
        f"Pérdida no soportada: {name}. "
        "Usa una de: bce, dice, bce_dice, combined, bce+dice, focal."
    )
