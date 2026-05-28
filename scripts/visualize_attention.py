"""
scripts/visualize_attention.py — Visualiza los mapas de atención
de la Attention U-Net.

Esto refuerza el análisis cualitativo (entregable #5) y muestra que
los Attention Gates están aprendiendo dónde mirar: las regiones de
vasos deberían "iluminarse" en los mapas ψ.

Mecánica:
  1. Registra forward hooks sobre cada AttentionGate del modelo.
  2. En cada hook, captura la salida del módulo `psi` (mapa α ∈ [0,1]).
  3. Para una imagen de test, ejecuta el modelo y grafica:
     imagen + GT + predicción + mapas de atención de cada nivel.

Uso:
  python scripts/visualize_attention.py \
      --config configs/train_attention_unet.yaml \
      --checkpoint outputs/attention_unet_drive/best_model.pt \
      --out_dir results/attention_maps \
      --n_images 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.attention_unet import AttentionUNet, AttentionGate
from data.dataset import DriveDataset
from data.transforms import ValTransform


def register_attention_hooks(model: AttentionUNet):
    """
    Registra hooks que capturan los mapas ψ de cada AttentionGate.
    Hookeamos directamente sobre `att.psi` (un Sequential que termina
    en Sigmoid), cuya salida es exactamente el mapa α ∈ [0,1].
    """
    captures = {}

    def make_hook(name):
        def hook(module, inp, output):
            captures[name] = output.detach().cpu()
        return hook

    handles = []
    for i, dec in enumerate(model.decoders):
        h = dec.att.psi.register_forward_hook(make_hook(f"level_{i}"))
        handles.append(h)
    return captures, handles


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    return img.permute(1, 2, 0).numpy().clip(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir",    type=str,
                        default="results/attention_maps")
    parser.add_argument("--n_images",   type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg["model"]["name"] != "attention_unet":
        raise ValueError("Este script requiere modelo attention_unet.")

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Modelo ─────────────────────────────────────────────────────────────
    model = AttentionUNet(
        in_channels=cfg["model"].get("in_channels", 3),
        out_channels=cfg["model"].get("out_channels", 1),
        base_channels=cfg["model"].get("base_channels", 64),
        depth=cfg["model"].get("depth", 4),
    )
    model.load_state_dict(torch.load(
        args.checkpoint, map_location=device, weights_only=True
    ))
    model = model.to(device).eval()

    captures, handles = register_attention_hooks(model)

    # ── Datos ──────────────────────────────────────────────────────────────
    img_size = cfg["data"].get("img_size", 512)
    use_clahe = cfg["data"].get("use_clahe", False)
    tf       = ValTransform(img_size=img_size, use_clahe=use_clahe)
    ds       = DriveDataset(
        cfg["data"].get("drive_root", "data/drive"),
        split="test", transform=tf,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    n_done = 0
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if n_done >= args.n_images:
                break

            img, mask = batch[0].to(device), batch[1]
            logits    = model(img)
            proba     = torch.sigmoid(logits).cpu().squeeze().numpy()
            pred_bin  = (proba > 0.5).astype(np.uint8)
            mask_np   = mask.squeeze().numpy().astype(np.uint8)
            img_show  = denormalize(img.squeeze())

            # captures ahora tiene un mapa por cada nivel del decoder
            n_levels = len(captures)
            fig, axes = plt.subplots(2, max(n_levels, 3),
                                     figsize=(4 * max(n_levels, 3), 8))
            if axes.ndim == 1:
                axes = axes[np.newaxis]

            axes[0, 0].imshow(img_show)
            axes[0, 0].set_title("Imagen", fontsize=11)
            axes[0, 0].axis("off")

            axes[0, 1].imshow(mask_np, cmap="gray")
            axes[0, 1].set_title("Ground truth", fontsize=11)
            axes[0, 1].axis("off")

            if axes.shape[1] >= 3:
                axes[0, 2].imshow(pred_bin, cmap="gray")
                axes[0, 2].set_title("Predicción", fontsize=11)
                axes[0, 2].axis("off")

            # Apagar paneles sobrantes de la fila 0
            for j in range(3, axes.shape[1]):
                axes[0, j].axis("off")

            # Fila 2: mapas de atención (uno por nivel del decoder)
            for j, lname in enumerate(sorted(captures.keys())):
                if j >= axes.shape[1]:
                    break
                amap = captures[lname].squeeze().numpy()   # (H, W)
                axes[1, j].imshow(img_show, alpha=0.5)
                axes[1, j].imshow(amap, cmap="jet", alpha=0.6,
                                   vmin=0, vmax=1)
                axes[1, j].set_title(f"Attention {lname}\n"
                                      f"(profundidad → superficie)",
                                      fontsize=10)
                axes[1, j].axis("off")

            for j in range(n_levels, axes.shape[1]):
                axes[1, j].axis("off")

            plt.suptitle(f"Mapas de atención — imagen #{idx+1}",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            fig_path = out_dir / f"attention_map_{idx+1:02d}.png"
            plt.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close()
            print(f"  ✓ {fig_path}")
            n_done += 1

    for h in handles:
        h.remove()

    print(f"\n✓ Mapas de atención guardados en: {out_dir}")


if __name__ == "__main__":
    main()
