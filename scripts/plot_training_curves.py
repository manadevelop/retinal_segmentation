"""
scripts/plot_training_curves.py — Grafica las curvas de entrenamiento
desde training_history.json.

Cualquier informe serio debe mostrar la convergencia: loss train/val
por época + F1 val. Esto va al apéndice o como figura principal.

Uso:
  python scripts/plot_training_curves.py \
      --history outputs/attention_unet_drive/training_history.json \
      --out_dir results/attention_unet_drive

  # Comparar dos experimentos en la misma figura:
  python scripts/plot_training_curves.py \
      --history outputs/attention_unet_drive/training_history.json \
               outputs/unet_base_drive/training_history.json \
      --labels  "Attention U-Net" "U-Net base" \
      --out_dir results/comparison
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, nargs="+", required=True,
                        help="Una o más rutas a training_history.json")
    parser.add_argument("--labels",  type=str, nargs="*", default=None,
                        help="Etiquetas (mismo orden que --history)")
    parser.add_argument("--out_dir", type=str, default="results/curves")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = []
    for path in args.history:
        with open(path) as f:
            histories.append(json.load(f))

    labels = args.labels or [Path(p).parent.name for p in args.history]
    assert len(labels) == len(histories), "Cantidad de labels ≠ historiales"

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Subplot 1: Loss train/val ─────────────────────────────────────────
    for i, (h, label) in enumerate(zip(histories, labels)):
        epochs = range(1, len(h["train_loss"]) + 1)
        c = colors[i % len(colors)]
        axes[0].plot(epochs, h["train_loss"], "-",  color=c,
                     alpha=0.6, label=f"{label} (train)")
        axes[0].plot(epochs, h["val_loss"],   "--", color=c,
                     lw=2, label=f"{label} (val)")
    axes[0].set_xlabel("Época", fontsize=11)
    axes[0].set_ylabel("Pérdida", fontsize=11)
    axes[0].set_title("Curva de pérdida", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # ── Subplot 2: F1 / Sens / Spec val ───────────────────────────────────
    for i, (h, label) in enumerate(zip(histories, labels)):
        epochs = range(1, len(h["val_f1"]) + 1)
        c = colors[i % len(colors)]
        axes[1].plot(epochs, h["val_f1"], "-", color=c, lw=2,
                     label=f"{label} F1")
        if "val_sensitivity" in h:
            axes[1].plot(epochs, h["val_sensitivity"], ":",
                         color=c, alpha=0.6, label=f"{label} Sens")

    axes[1].set_xlabel("Época", fontsize=11)
    axes[1].set_ylabel("Métrica", fontsize=11)
    axes[1].set_title("Métricas de validación", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.0)

    plt.suptitle("Curvas de entrenamiento", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = out_dir / "training_curves.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fig_path}")


if __name__ == "__main__":
    main()
