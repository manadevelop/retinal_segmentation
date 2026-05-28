"""Métricas robustas para segmentación binaria de vasos retinianos."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_metrics_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Calcula métricas por batch. Si `valid_mask` existe, evalúa solo dentro del FOV."""
    preds = preds.detach().float()
    targets = targets.detach().float()
    preds_bin = (preds > threshold).float()

    if valid_mask is not None:
        vm = (valid_mask.detach().float() > 0.5)
        p = preds_bin[vm].cpu().numpy()
        t = targets[vm].cpu().numpy()
        s = preds[vm].cpu().numpy()
    else:
        p = preds_bin.reshape(-1).cpu().numpy()
        t = targets.reshape(-1).cpu().numpy()
        s = preds.reshape(-1).cpu().numpy()

    if t.size == 0:
        return {k: float("nan") for k in ["sensibilidad", "especificidad", "precision", "f1", "iou", "accuracy", "auc_roc"]}

    t = (t > 0.5).astype(np.uint8)
    p = (p > 0.5).astype(np.uint8)

    TP = float(((p == 1) & (t == 1)).sum())
    TN = float(((p == 0) & (t == 0)).sum())
    FP = float(((p == 1) & (t == 0)).sum())
    FN = float(((p == 0) & (t == 1)).sum())

    sensibilidad = TP / (TP + FN + smooth)
    especificidad = TN / (TN + FP + smooth)
    precision = TP / (TP + FP + smooth)
    f1 = 2 * TP / (2 * TP + FP + FN + smooth)
    iou = TP / (TP + FP + FN + smooth)
    accuracy = (TP + TN) / (TP + TN + FP + FN + smooth)

    auc_roc = float("nan")
    if np.unique(t).size == 2:
        try:
            auc_roc = float(roc_auc_score(t, s))
        except ValueError:
            pass

    return {
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "precision": precision,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "auc_roc": auc_roc,
    }


def compute_metrics_dataset(
    all_preds: List[np.ndarray],
    all_targets: List[np.ndarray],
    all_valid_masks: Optional[List[np.ndarray]] = None,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    metrics_list = []
    if all_valid_masks is None:
        all_valid_masks = [None] * len(all_preds)
    for pred, target, vm in zip(all_preds, all_targets, all_valid_masks):
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
        target_t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
        vm_t = torch.from_numpy(vm).unsqueeze(0).unsqueeze(0) if vm is not None else None
        metrics_list.append(compute_metrics_batch(pred_t, target_t, vm_t, threshold, smooth))

    result = {}
    for k in metrics_list[0].keys():
        vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
        result[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        result[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
    return result


def print_metrics(metrics: Dict, prefix: str = "") -> None:
    p = f"[{prefix}] " if prefix else ""
    print(f"\n{p}{'='*50}")
    for label, key in [
        ("Sensibilidad ", "sensibilidad"),
        ("Especificidad", "especificidad"),
        ("Precision    ", "precision"),
        ("F1 (Dice)    ", "f1"),
        ("AUC-ROC      ", "auc_roc"),
        ("IoU          ", "iou"),
        ("Accuracy     ", "accuracy"),
    ]:
        if f"{key}_mean" in metrics:
            m = metrics.get(f"{key}_mean", float("nan"))
            s = metrics.get(f"{key}_std", float("nan"))
            print(f"{p}{label}: {m:.4f} ± {s:.4f}")
        else:
            print(f"{p}{label}: {metrics.get(key, float('nan')):.4f}")
    print(f"{p}{'='*50}\n")
