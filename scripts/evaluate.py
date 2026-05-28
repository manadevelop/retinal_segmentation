"""
scripts/evaluate.py — Evaluación completa y visualizaciones para segmentación retinal.

Genera:
  - Métricas en test (sensibilidad, especificidad, F1, AUC-ROC, IoU)
  - Visualizaciones de predicciones vs ground truth
  - Análisis de fallos (capilares finos vs arterias grandes)
  - Curvas ROC

Uso:
  python scripts/evaluate.py \
    --config configs/train_attention_unet.yaml \
    --checkpoint outputs/attention_unet_drive/best_model.pt \
    --out_dir results/attention_unet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.attention_unet import AttentionUNet
from models.unet import UNet
from data.dataset import DriveDataset, ChaseDB1Dataset
from data.transforms import ValTransform
from utils.metrics import compute_metrics_batch, print_metrics


def load_model(cfg, checkpoint, device):
    name = cfg["model"]["name"]
    args = dict(
        in_channels=cfg["model"].get("in_channels", 3),
        out_channels=cfg["model"].get("out_channels", 1),
        base_channels=cfg["model"].get("base_channels", 64),
        depth=cfg["model"].get("depth", 4),
    )
    model = AttentionUNet(**args) if name == "attention_unet" else UNet(**args)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    return img.permute(1, 2, 0).numpy().clip(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir",    type=str, default="results/eval")
    parser.add_argument("--dataset",    type=str, default="drive")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model    = load_model(cfg, args.checkpoint, device)
    img_size = cfg["data"].get("img_size", 512)
    tf       = ValTransform(img_size=img_size)

    if args.dataset == "drive":
        drive_root = cfg["data"].get("drive_root", "data/drive")
        try:
            ds = DriveDataset(drive_root, split="test", transform=tf)
        except RuntimeError as e:
            print("⚠ DRIVE test no tiene anotaciones manuales públicas o no están montadas.")
            print(f"  Detalle: {e}")
            print("  Usando DRIVE/training como holdout evaluable para generar métricas locales.")
            ds = DriveDataset(drive_root, split="train", transform=tf)
    else:
        ds = ChaseDB1Dataset(
            cfg["data"].get("chase_root", "data/chase_db1"),
            split="all", transform=tf
        )

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    print(f"Evaluando {len(ds)} imágenes...")

    all_preds, all_targets, all_imgs, all_fovs = [], [], [], []
    accum = {k: [] for k in
             ("sensibilidad", "especificidad", "precision",
              "f1", "iou", "accuracy", "auc_roc")}

    with torch.no_grad():
        for batch in loader:
            img, mask = batch[0].to(device), batch[1].to(device)
            fov = batch[2].to(device) if len(batch) > 2 else None
            logits    = model(img)
            proba     = torch.sigmoid(logits)

            m = compute_metrics_batch(proba, mask, valid_mask=fov)
            for k in accum:
                accum[k].append(m[k])

            all_preds.append(proba.squeeze().cpu().numpy())
            all_targets.append(mask.squeeze().cpu().numpy())
            all_fovs.append(fov.squeeze().cpu().numpy() if fov is not None else np.ones_like(mask.squeeze().cpu().numpy()))
            all_imgs.append(img.squeeze().cpu())

    # ── Métricas finales (promedio + std por imagen) ──────────────────────
    metrics = {}
    for k, vals in accum.items():
        arr = np.array([v for v in vals if not np.isnan(v)])
        metrics[k]            = float(arr.mean()) if arr.size else float("nan")
        metrics[f"{k}_std"]   = float(arr.std())  if arr.size else float("nan")
    print_metrics(metrics, prefix=cfg["model"]["name"])

    # Para mantener compatibilidad con código que busca f1_l[i]
    f1_l   = accum["f1"]
    sens_l = accum["sensibilidad"]

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Visualizaciones: predicciones vs ground truth ─────────────────────
    n_vis = min(4, len(all_imgs))
    fig, axes = plt.subplots(n_vis, 4, figsize=(16, n_vis * 4))
    if n_vis == 1:
        axes = axes[np.newaxis]

    for i in range(n_vis):
        pred_bin = (all_preds[i] > 0.5).astype(np.float32)

        axes[i, 0].imshow(denormalize(all_imgs[i]))
        axes[i, 0].set_title("Imagen original", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(all_targets[i], cmap="gray")
        axes[i, 1].set_title("Ground Truth", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(all_preds[i], cmap="hot")
        axes[i, 2].set_title("Probabilidad predicha", fontsize=9)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred_bin, cmap="gray")
        axes[i, 3].set_title(
            f"Predicción binaria\nF1={f1_l[i]:.3f}", fontsize=9
        )
        axes[i, 3].axis("off")

    plt.suptitle(
        f"Segmentación de vasos retinianos — {cfg['model']['name']}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "predictions.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ predictions.png")

    # ── Curva ROC ──────────────────────────────────────────────────────────
    preds_flat = np.concatenate([p[f > 0.5].flatten() for p, f in zip(all_preds, all_fovs)])
    targets_flat = np.concatenate([t[f > 0.5].flatten() for t, f in zip(all_targets, all_fovs)])

    if np.unique(targets_flat.astype(int)).size < 2:
        print("  ⚠ ROC no generado: ground truth contiene una sola clase dentro del FOV.")
        fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), float("nan")
    else:
        fpr, tpr, _ = roc_curve(targets_flat.astype(int), preds_flat)
        roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#F44336", lw=2,
            label=f"{cfg['model']['name']} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
    ax.set_title("Curva ROC — Segmentación de vasos retinianos", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ roc_curve.png")

    # ── Análisis de fallos ─────────────────────────────────────────────────
    # Ordenar imágenes por F1 ascendente (peores primero)
    indices_sorted = np.argsort(f1_l)[:5]

    fig, axes = plt.subplots(len(indices_sorted), 4,
                             figsize=(16, len(indices_sorted) * 4))
    if len(indices_sorted) == 1:
        axes = axes[np.newaxis]

    for rank, idx in enumerate(indices_sorted):
        pred_bin = (all_preds[idx] > 0.5).astype(np.float32)

        # Diferencia: FP en rojo, FN en azul
        diff = np.zeros((*pred_bin.shape, 3))
        fp   = (pred_bin == 1) & (all_targets[idx] == 0)
        fn   = (pred_bin == 0) & (all_targets[idx] == 1)
        diff[fp] = [1, 0, 0]   # Falsos Positivos: rojo
        diff[fn] = [0, 0, 1]   # Falsos Negativos: azul
        diff[(pred_bin == 1) & (all_targets[idx] == 1)] = [0, 1, 0]  # TP: verde

        axes[rank, 0].imshow(denormalize(all_imgs[idx]))
        axes[rank, 0].set_title(f"Imagen #{idx+1}", fontsize=9)
        axes[rank, 0].axis("off")

        axes[rank, 1].imshow(all_targets[idx], cmap="gray")
        axes[rank, 1].set_title("Ground Truth", fontsize=9)
        axes[rank, 1].axis("off")

        axes[rank, 2].imshow(pred_bin, cmap="gray")
        axes[rank, 2].set_title(f"Predicción (F1={f1_l[idx]:.3f})", fontsize=9)
        axes[rank, 2].axis("off")

        axes[rank, 3].imshow(diff)
        axes[rank, 3].set_title(
            "Verde=TP, Rojo=FP, Azul=FN", fontsize=9
        )
        axes[rank, 3].axis("off")

    plt.suptitle(
        "Análisis de fallos — 5 peores segmentaciones",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "failure_analysis.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ failure_analysis.png")

    print(f"\n✓ Evaluación completa guardada en: {out_dir}")


if __name__ == "__main__":
    main()
