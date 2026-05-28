"""
scripts/inter_annotator_stare.py — Análisis de concordancia entre anotadores
sobre STARE.

STARE incluye dos conjuntos de anotaciones manuales (AH y VK) sobre las
mismas 20 imágenes, lo que permite estimar un "upper bound" humano:
si dos expertos discrepan en X% de los píxeles, ningún modelo debería
considerarse "mejor que el humano" sin discutir este límite.

Métricas reportadas (AH como referencia, VK como predicción):
  - Acuerdo a nivel píxel (accuracy)
  - Dice / F1
  - IoU
  - Coeficiente kappa de Cohen (corrige acuerdo por azar)
  - Comparación opcional: tu modelo vs cada anotador

Uso:
  python scripts/inter_annotator_stare.py \
      --stare_root data/stare \
      --out_dir results/inter_annotator
  # (opcional) comparar modelo entrenado vs los dos anotadores:
  python scripts/inter_annotator_stare.py \
      --stare_root data/stare \
      --out_dir results/inter_annotator \
      --config configs/train_attention_unet.yaml \
      --checkpoint outputs/attention_unet_drive/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_mask(path: Path) -> np.ndarray:
    """Carga una máscara binaria desde PPM/PNG."""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > 127).astype(np.uint8)


def pairwise_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Métricas píxel a píxel entre dos máscaras binarias (a vs b)."""
    a = a.astype(np.uint8).flatten()
    b = b.astype(np.uint8).flatten()
    smooth = 1e-6

    TP = float(((a == 1) & (b == 1)).sum())
    TN = float(((a == 0) & (b == 0)).sum())
    FP = float(((a == 0) & (b == 1)).sum())   # b dice vaso, a dice fondo
    FN = float(((a == 1) & (b == 0)).sum())
    N  = TP + TN + FP + FN

    accuracy = (TP + TN) / (N + smooth)
    dice     = 2 * TP / (2 * TP + FP + FN + smooth)
    iou      = TP / (TP + FP + FN + smooth)

    # Cohen's kappa: corrige por acuerdo esperado al azar
    p_obs = accuracy
    p_a1  = (TP + FN) / (N + smooth)   # frecuencia de "vaso" en a
    p_b1  = (TP + FP) / (N + smooth)   # frecuencia de "vaso" en b
    p_exp = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    kappa = (p_obs - p_exp) / (1 - p_exp + smooth)

    return {
        "accuracy": accuracy,
        "dice":     dice,
        "iou":      iou,
        "kappa":    kappa,
        "n_pixels_a": int(TP + FN),
        "n_pixels_b": int(TP + FP),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stare_root", type=str, default="data/stare")
    parser.add_argument("--out_dir",    type=str,
                        default="results/inter_annotator")
    parser.add_argument("--config",     type=str, default=None,
                        help="(opcional) config para comparar modelo vs anotadores")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="(opcional) checkpoint del modelo")
    args = parser.parse_args()

    stare_root = Path(args.stare_root)
    img_dir    = stare_root / "images"
    ah_dir     = stare_root / "labels-ah"
    vk_dir     = stare_root / "labels-vk"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(img_dir.glob("*.ppm")) +
        list(img_dir.glob("*.png"))
    )
    if not images:
        raise RuntimeError(f"No hay imágenes en {img_dir}")

    print(f"Procesando {len(images)} imágenes de STARE...")

    per_image = []
    accum = {"accuracy": [], "dice": [], "iou": [], "kappa": []}

    for img_path in images:
        stem = img_path.stem
        ah_cand = list(ah_dir.glob(f"{stem}*")) + list(ah_dir.glob(f"{stem}.*"))
        vk_cand = list(vk_dir.glob(f"{stem}*")) + list(vk_dir.glob(f"{stem}.*"))

        if not ah_cand or not vk_cand:
            print(f"  ⚠ Falta anotador para {stem}, salteando")
            continue

        ah_mask = load_mask(ah_cand[0])
        vk_mask = load_mask(vk_cand[0])

        # Asegurar mismo tamaño (debería siempre coincidir en STARE)
        if ah_mask.shape != vk_mask.shape:
            print(f"  ⚠ Tamaños distintos para {stem}, saltando")
            continue

        m = pairwise_metrics(ah_mask, vk_mask)
        m["image"] = stem
        per_image.append(m)
        for k in accum:
            accum[k].append(m[k])

    # ── Resumen ────────────────────────────────────────────────────────────
    summary = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in accum.items()
    }
    n = len(per_image)

    print(f"\n{'='*60}")
    print(f"Concordancia AH vs VK sobre {n} imágenes de STARE")
    print(f"{'='*60}")
    print(f"  Accuracy    : {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
    print(f"  Dice / F1   : {summary['dice']['mean']:.4f} ± {summary['dice']['std']:.4f}")
    print(f"  IoU         : {summary['iou']['mean']:.4f} ± {summary['iou']['std']:.4f}")
    print(f"  Kappa Cohen : {summary['kappa']['mean']:.4f} ± {summary['kappa']['std']:.4f}")
    print(f"{'='*60}\n")

    output = {
        "n_images": n,
        "summary":  summary,
        "per_image": per_image,
    }

    # ── Si se proporciona modelo, comparar contra ambos anotadores ────────
    if args.config and args.checkpoint:
        import yaml
        import torch
        from torch.utils.data import DataLoader
        from models.attention_unet import AttentionUNet
        from models.unet import UNet
        from data.dataset import StareDataset
        from data.transforms import ValTransform

        print("Comparando modelo contra ambos anotadores...")
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        device   = "cuda" if torch.cuda.is_available() else "cpu"
        img_size = cfg["data"].get("img_size", 512)
        use_clahe = cfg["data"].get("use_clahe", False)
        tf       = ValTransform(img_size=img_size, use_clahe=use_clahe)

        name = cfg["model"]["name"]
        margs = dict(
            in_channels=cfg["model"].get("in_channels", 3),
            out_channels=cfg["model"].get("out_channels", 1),
            base_channels=cfg["model"].get("base_channels", 64),
            depth=cfg["model"].get("depth", 4),
        )
        model = AttentionUNet(**margs) if name == "attention_unet" else UNet(**margs)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                         weights_only=True))
        model = model.to(device).eval()

        model_vs_ah, model_vs_vk = [], []
        for annotator, lst in [("ah", model_vs_ah), ("vk", model_vs_vk)]:
            ds = StareDataset(args.stare_root, annotator=annotator, transform=tf)
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            with torch.no_grad():
                for batch in loader:
                    img, mask = batch[0].to(device), batch[1]
                    proba = torch.sigmoid(model(img)).cpu().squeeze().numpy()
                    pred  = (proba > 0.5).astype(np.uint8)
                    gt    = mask.squeeze().numpy().astype(np.uint8)
                    lst.append(pairwise_metrics(gt, pred))

        for tag, lst in [("model_vs_ah", model_vs_ah),
                         ("model_vs_vk", model_vs_vk)]:
            s = {k: {"mean": float(np.mean([m[k] for m in lst])),
                      "std":  float(np.std([m[k] for m in lst]))}
                 for k in ("accuracy", "dice", "iou", "kappa")}
            output[tag] = s
            print(f"  {tag}: Dice={s['dice']['mean']:.4f} | "
                  f"Kappa={s['kappa']['mean']:.4f}")

    # ── Guardar JSON ───────────────────────────────────────────────────────
    out_json = out_dir / "inter_annotator_results.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ {out_json}")

    # ── Figura: distribución de Dice y Kappa ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dices  = [m["dice"]  for m in per_image]
    kappas = [m["kappa"] for m in per_image]

    axes[0].hist(dices, bins=10, color="#3498DB",
                 alpha=0.7, edgecolor="black")
    axes[0].axvline(np.mean(dices), color="red", linestyle="--", lw=2,
                    label=f"Media = {np.mean(dices):.3f}")
    axes[0].set_xlabel("Dice / F1 (AH vs VK)", fontsize=11)
    axes[0].set_ylabel("Número de imágenes", fontsize=11)
    axes[0].set_title("Concordancia AH vs VK — Dice", fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(kappas, bins=10, color="#9B59B6",
                 alpha=0.7, edgecolor="black")
    axes[1].axvline(np.mean(kappas), color="red", linestyle="--", lw=2,
                    label=f"Media = {np.mean(kappas):.3f}")
    axes[1].set_xlabel("Cohen's Kappa (AH vs VK)", fontsize=11)
    axes[1].set_ylabel("Número de imágenes", fontsize=11)
    axes[1].set_title("Concordancia AH vs VK — Kappa", fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Concordancia entre anotadores en STARE",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = out_dir / "agreement_distribution.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fig_path}")

    # ── Discusión guía para el informe ─────────────────────────────────────
    discuss = f"""# Concordancia entre anotadores — STARE

Acuerdo entre los dos expertos sobre {n} imágenes:

| Métrica  | Media ± std |
|----------|-------------|
| Accuracy | {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f} |
| Dice/F1  | {summary['dice']['mean']:.4f} ± {summary['dice']['std']:.4f} |
| IoU      | {summary['iou']['mean']:.4f} ± {summary['iou']['std']:.4f} |
| Kappa    | {summary['kappa']['mean']:.4f} ± {summary['kappa']['std']:.4f} |

## Interpretación

El Dice promedio entre AH y VK ({summary['dice']['mean']:.3f}) es el
"techo humano" práctico: ningún modelo puede superarlo de forma
significativa sin sobreajustar a las idiosincrasias de un solo
anotador.

Kappa de Cohen entre {summary['kappa']['mean']:.3f} es muy alto
(generalmente >0.8 se considera "casi perfecto"), pero queda claramente
por debajo de 1.0. Esto significa que incluso entre expertos hay
desacuerdos sistemáticos — típicamente en los capilares más finos y
en los bordes de los vasos grandes, donde la decisión píxel-a-píxel
es subjetiva.

Comparación con tu modelo: si tu modelo alcanza un F1 de ~0.80 contra
AH y ~0.80 contra VK pero solo ~{summary['dice']['mean']:.2f} contra el
ensemble, está dentro del rango humano. Reportar ambas comparaciones
es buena práctica.
"""
    with open(out_dir / "discussion.md", "w") as f:
        f.write(discuss)
    print(f"  ✓ {out_dir / 'discussion.md'}")


if __name__ == "__main__":
    main()
