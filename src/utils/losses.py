"""
Funciones de pérdida para segmentación binaria de vasos retinianos.

Soporta:
- BCEWithLogitsLoss
- DiceLoss
- BCE + Dice
- FocalLoss

Incluye manejo robusto de pos_weight para configuraciones YAML como:

    pos_weight: 5.0

o:

    pos_weight:
      enabled: true
      value: 5.0
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_pos_weight(pos_weight: Any) -> Optional[float]:
    """
    Convierte pos_weight desde YAML/Python a float o None.

    Casos soportados:
    - None
    - 5
    - 5.0
    - "5.0"
    - {"enabled": true, "value": 5.0}
    - {"enabled": false, "value": 5.0}
    - {"value": 5.0}
    - {"weight": 5.0}
    - {"pos_weight": 5.0}
    """

    if pos_weight is None:
        return None

    if isinstance(pos_weight, bool):
        return None

    if isinstance(pos_weight, (int, float)):
        value = float(pos_weight)
        return value if value > 0 else None

    if isinstance(pos_weight, str):
        value = float(pos_weight)
        return value if value > 0 else None

    if isinstance(pos_weight, torch.Tensor):
        if pos_weight.numel() == 0:
            return None
        value = float(pos_weight.detach().cpu().flatten()[0].item())
        return value if value > 0 else None

    if isinstance(pos_weight, dict):
        enabled = pos_weight.get("enabled", True)

        if enabled is False:
            return None

        for key in ("value", "weight", "pos_weight", "positive_weight"):
            if key in pos_weight and pos_weight[key] is not None:
                value = float(pos_weight[key])
                return value if value > 0 else None

        return None

    if isinstance(pos_weight, (list, tuple)):
        if len(pos_weight) == 0:
            return None
        value = float(pos_weight[0])
        return value if value > 0 else None

    raise TypeError(
        f"Formato no soportado para pos_weight: {type(pos_weight)}. "
        "Usa un número o un diccionario con {'enabled': true, 'value': número}."
    )


class BCELoss(nn.Module):
    """
    Binary Cross Entropy con logits.

    Compatible con CPU/GPU. Si pos_weight existe, se registra como buffer
    y se mueve automáticamente al device de los logits durante forward.
    """

    def __init__(self, pos_weight: Any = None):
        super().__init__()

        parsed = _parse_pos_weight(pos_weight)

        if parsed is None:
            self.register_buffer("pos_weight", torch.empty(0, dtype=torch.float32))
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor([parsed], dtype=torch.float32),
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        if self.pos_weight.numel() > 0:
            pos_weight = self.pos_weight.to(
                device=logits.device,
                dtype=logits.dtype,
            )
        else:
            pos_weight = None

        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )


class DiceLoss(nn.Module):
    """
    Dice Loss para segmentación binaria.

    Recibe logits y aplica sigmoid internamente.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Pérdida combinada BCE + Dice.

    BCE estabiliza el aprendizaje píxel a píxel.
    Dice favorece el solapamiento de estructuras delgadas.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Any = None,
        smooth: float = 1.0,
    ):
        super().__init__()

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

        self.bce = BCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss binaria con logits.

    Útil para desbalance fuerte foreground/background.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)

        alpha_pos = torch.tensor(
            self.alpha,
            device=logits.device,
            dtype=logits.dtype,
        )
        alpha_neg = torch.tensor(
            1.0 - self.alpha,
            device=logits.device,
            dtype=logits.dtype,
        )

        alpha_weight = torch.where(targets == 1, alpha_pos, alpha_neg)
        focal_weight = (1.0 - pt).pow(self.gamma)

        loss = alpha_weight * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


def _get_from_dict(cfg: dict, keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def get_loss(loss_cfg: Any, training_cfg: Optional[dict] = None) -> nn.Module:
    """
    Factory de pérdidas.

    Soporta llamadas como:

        get_loss("bce")
        get_loss({"name": "bce_dice", "pos_weight": {"enabled": true, "value": 5.0}})
        get_loss(cfg["training"]["loss"], cfg["training"])

    Nombres soportados:
    - bce
    - dice
    - bce_dice
    - combined
    - bce+dice
    - focal
    """

    training_cfg = training_cfg or {}

    if isinstance(loss_cfg, str):
        name = loss_cfg
        cfg = {}
    elif isinstance(loss_cfg, dict):
        cfg = loss_cfg
        name = cfg.get("name", cfg.get("type", "bce_dice"))
    else:
        raise TypeError(
            f"loss_cfg debe ser str o dict, pero llegó {type(loss_cfg)}"
        )

    name = str(name).lower().strip()

    pos_weight = _get_from_dict(
        cfg,
        ("pos_weight", "positive_weight", "class_weight"),
        default=None,
    )

    if pos_weight is None and isinstance(training_cfg, dict):
        pos_weight = _get_from_dict(
            training_cfg,
            ("pos_weight", "positive_weight", "class_weight"),
            default=None,
        )

    bce_weight = float(
        _get_from_dict(cfg, ("bce_weight", "lambda_bce"), default=0.5)
    )

    dice_weight = float(
        _get_from_dict(cfg, ("dice_weight", "lambda_dice"), default=0.5)
    )

    smooth = float(
        _get_from_dict(cfg, ("smooth", "eps"), default=1.0)
    )

    if name in {"bce", "binary_cross_entropy", "bcewithlogits"}:
        return BCELoss(pos_weight=pos_weight)

    if name in {"dice", "dice_loss"}:
        return DiceLoss(smooth=smooth)

    if name in {
        "bce_dice",
        "combined",
        "bce+dice",
        "dice_bce",
        "bce-dice",
    }:
        return BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
            smooth=smooth,
        )

    if name in {"focal", "focal_loss"}:
        alpha = float(_get_from_dict(cfg, ("alpha",), default=0.25))
        gamma = float(_get_from_dict(cfg, ("gamma",), default=2.0))
        return FocalLoss(alpha=alpha, gamma=gamma)

    raise ValueError(
        f"Pérdida no soportada: {name}. "
        "Usa una de: bce, dice, bce_dice, combined, bce+dice, focal."
    )