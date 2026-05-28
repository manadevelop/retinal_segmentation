"""
scripts/vessel_thickness_analysis.py — Análisis estratificado por grosor de vaso.

Aborda el entregable #5 del examen, que pide identificar qué tipos de
vasos (capilares finos vs. arterias grandes) son más difíciles de segmentar
y explicar por qué desde una perspectiva arquitectónica.

Metodología:
  1. Para cada imagen, sobre la máscara ground truth aplicamos la
     transformada de distancia (distance_transform_edt). El valor en cada
     píxel-vaso es la distancia mínima al fondo, ≈ radio local del vaso.
  2. Definimos tres categorías por radio (en píxeles, sobre img_size=512):
       FINO   : radio ≤ 2  (capilares)
       MEDIO  : 2 < radio ≤ 4
       GRUESO : radio > 4  (arterias grandes, venas principales)
  3. Calculamos la sensibilidad (recall) del modelo por categoría:
       sens_cat = #(TP píxeles en categoría) / #(píxeles GT en categoría)
  4. Reportamos la sensibilidad por categoría y por imagen,
     más una figura comparativa.

Uso:
  python scripts/vessel_thickness_analysis.py \
      --config configs/train_attention_unet.yaml \
      --checkpoint outputs/attention_unet_drive/best_model.pt \
      --out_dir results/vessel_thickness \
      --dataset drive
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.attention_unet import AttentionUNet
from models.unet import UNet
from data.dataset import DriveDataset, ChaseDB1Dataset
from data.transforms import ValTransform


# Umbrales de radio (en píxeles) que definen las categorías de grosor
THRESHOLDS = {"fino": 2, "medio": 4}   # > 4 => grueso


def load_model(cfg, checkpoint, device):
    args = dict(
        in_channels=cfg["model"].get("in_channels", 3),
        out_channels=cfg["model"].get("out_channels", 1),
        base_channels=cfg["model"].get("base_channels", 64),
        depth=cfg["model"].get("depth", 4),
    )
    name  = cfg["model"]["name"]
    model = AttentionUNet(**args) if name == "attention_unet" else UNet(**args)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def stratify_by_thickness(mask: np.ndarray) -> dict:
    """
    Devuelve máscaras booleanas por categoría de grosor.

    Parámetros
    ----------
    mask : (H, W) binaria {0,1}

    Devuelve
    --------
    {"fino": bool_mask, "medio": bool_mask, "grueso": bool_mask}
    """
    # Distancia desde cada píxel-vaso al fondo más cercano (≈ radio local)
    radii = distance_transform_edt(mask > 0)

    fino   = (mask > 0) & (radii <= THRESHOLDS["fino"])
    medio  = (mask > 0) & (radii > THRESHOLDS["fino"]) & (radii <= THRESHOLDS["medio"])
    grueso = (mask > 0) & (radii >  THRESHOLDS["medio"])

    return {"fino": fino, "medio": medio, "grueso": grueso}


def sensitivity_per_category(pred_bin: np.ndarray,
                              cats: dict) -> dict:
    """Sensibilidad por categoría: TP/(TP+FN) restringido a cada subconjunto."""
    out = {}
    for name, mask_cat in cats.items():
        gt_n = int(mask_cat.sum())
        if gt_n == 0:
            out[name] = {"sensitivity": float("nan"), "n_pixels": 0}
            continue
        tp = int(((pred_bin > 0) & mask_cat).sum())
        out[name] = {
            "sensitivity": tp / gt_n,
            "n_pixels":    gt_n,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir",    type=str,
                        default="results/vessel_thickness")
    parser.add_argument("--dataset",    type=str, default="drive")
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model    = load_model(cfg, args.checkpoint, device)
    img_size = cfg["data"].get("img_size", 512)
    use_clahe = cfg["data"].get("use_clahe", False)
    tf       = ValTransform(img_size=img_size, use_clahe=use_clahe)

    if args.dataset == "drive":
        drive_root = cfg["data"].get("drive_root", "data/drive")
        try:
            ds = DriveDataset(drive_root, split="test", transform=tf)
        except RuntimeError as e:
            print("⚠ DRIVE test no tiene anotaciones manuales; usando DRIVE/training para análisis local.")
            print(f"  Detalle: {e}")
            ds = DriveDataset(drive_root, split="train", transform=tf)
    else:
        ds = ChaseDB1Dataset(
            cfg["data"].get("chase_root", "data/chase_db1"),
            split="all", transform=tf
        )

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Analizando {len(ds)} imágenes por grosor de vaso...")

    # Acumular sensibilidad por categoría
    cat_results = {"fino": [], "medio": [], "grueso": []}
    per_image_rows = []   # filas para CSV / tabla

    with torch.no_grad():
        for i, batch in enumerate(loader):
            img, mask = batch[0].to(device), batch[1]
            logits    = model(img)
            proba     = torch.sigmoid(logits).cpu().squeeze().numpy()
            pred_bin  = (proba > args.threshold).astype(np.uint8)
            mask_np   = mask.squeeze().numpy().astype(np.uint8)

            cats   = stratify_by_thickness(mask_np)
            sens   = sensitivity_per_category(pred_bin, cats)

            row = {"image_idx": i}
            for cat in ("fino", "medio", "grueso"):
                row[f"sens_{cat}"]   = sens[cat]["sensitivity"]
                row[f"npix_{cat}"]   = sens[cat]["n_pixels"]
                if not np.isnan(sens[cat]["sensitivity"]):
                    cat_results[cat].append(sens[cat]["sensitivity"])
            per_image_rows.append(row)

    # Estadísticas globales
    summary = {}
    for cat, vals in cat_results.items():
        arr = np.array(vals)
        summary[cat] = {
            "sens_mean":   float(arr.mean())   if arr.size else float("nan"),
            "sens_std":    float(arr.std())    if arr.size else float("nan"),
            "sens_median": float(np.median(arr)) if arr.size else float("nan"),
            "n_images":    int(arr.size),
        }

    # ── Imprimir resumen ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Categoría':<10} {'#imgs':>6} {'Sens. media':>15} {'± std':>10} {'Mediana':>10}")
    print(f"{'-'*60}")
    for cat in ("fino", "medio", "grueso"):
        s = summary[cat]
        print(f"{cat:<10} {s['n_images']:>6} "
              f"{s['sens_mean']:>15.4f} "
              f"{s['sens_std']:>10.4f} "
              f"{s['sens_median']:>10.4f}")
    print(f"{'='*60}\n")

    # ── Guardar JSON con todo ──────────────────────────────────────────────
    output = {
        "thresholds":   THRESHOLDS,
        "summary":      summary,
        "per_image":    per_image_rows,
        "model":        cfg["model"]["name"],
        "dataset":      args.dataset,
        "checkpoint":   args.checkpoint,
    }
    out_json = out_dir / "vessel_thickness_results.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ {out_json}")

    # ── Figura comparativa: bar chart con error bars ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    cats   = ["fino", "medio", "grueso"]
    labels = ["Fino\n(capilares,\nradio ≤ 2 px)",
              "Medio\n(2 < radio ≤ 4)",
              "Grueso\n(arterias,\nradio > 4 px)"]
    means  = [summary[c]["sens_mean"] for c in cats]
    stds   = [summary[c]["sens_std"]  for c in cats]
    colors = ["#E74C3C", "#F39C12", "#2ECC71"]

    bars = ax.bar(labels, means, yerr=stds, capsize=8,
                  color=colors, alpha=0.85, edgecolor="black")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Sensibilidad (Recall)", fontsize=12)
    ax.set_title(f"Sensibilidad por grosor de vaso — {cfg['model']['name']} "
                 f"({args.dataset.upper()})",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / "sensitivity_by_thickness.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fig_path}")

    # ── Visualización: imagen con vasos coloreados por categoría ──────────
    # Tomamos la primera imagen y mostramos GT estratificada
    with torch.no_grad():
        batch = next(iter(loader))
        img, mask = batch[0].to(device), batch[1]
        logits   = model(img)
        proba    = torch.sigmoid(logits).cpu().squeeze().numpy()
        pred_bin = (proba > args.threshold).astype(np.uint8)
        mask_np  = mask.squeeze().numpy().astype(np.uint8)
        cats     = stratify_by_thickness(mask_np)

    # Composite RGB: rojo=fino, amarillo=medio, verde=grueso
    stratified = np.zeros((*mask_np.shape, 3), dtype=np.float32)
    stratified[cats["fino"]]   = [1.0, 0.0, 0.0]
    stratified[cats["medio"]]  = [1.0, 0.7, 0.0]
    stratified[cats["grueso"]] = [0.0, 1.0, 0.0]

    # Errores: rojo donde el modelo erró (FN)
    fn_strat = np.zeros((*mask_np.shape, 3), dtype=np.float32)
    for cat_name, color in [("fino", [1.0, 0.0, 0.0]),
                             ("medio", [1.0, 0.7, 0.0]),
                             ("grueso", [0.0, 1.0, 0.0])]:
        fn_mask = cats[cat_name] & (pred_bin == 0)
        fn_strat[fn_mask] = color

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask_np, cmap="gray")
    axes[0].set_title("Ground truth", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(stratified)
    axes[1].set_title("GT estratificado\n"
                      "(rojo=fino, ámbar=medio, verde=grueso)",
                      fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(fn_strat)
    axes[2].set_title("Falsos negativos\npor categoría", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    fig2_path = out_dir / "thickness_visualization.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fig2_path}")

    # ── Interpretación arquitectónica (texto guía para el informe) ────────
    interp_path = out_dir / "interpretation.md"
    sens_fino, sens_grueso = summary["fino"]["sens_mean"], summary["grueso"]["sens_mean"]
    gap = sens_grueso - sens_fino

    interp = f"""# Interpretación arquitectónica — Sensibilidad por grosor

## Resultados numéricos

| Categoría | Definición (radio en px sobre {img_size}×{img_size}) | Sensibilidad media |
|-----------|------------------------------------------------------|--------------------|
| Fino      | radio ≤ {THRESHOLDS['fino']}                         | {summary['fino']['sens_mean']:.4f} ± {summary['fino']['sens_std']:.4f} |
| Medio     | {THRESHOLDS['fino']} < radio ≤ {THRESHOLDS['medio']} | {summary['medio']['sens_mean']:.4f} ± {summary['medio']['sens_std']:.4f} |
| Grueso    | radio > {THRESHOLDS['medio']}                        | {summary['grueso']['sens_mean']:.4f} ± {summary['grueso']['sens_std']:.4f} |

Brecha grueso − fino: {gap:+.4f}

## Explicación arquitectónica

1. **Receptive field y downsampling.** U-Net hace 4 maxpoolings (factor 16
   total). Un capilar de radio 1-2 px en la entrada queda con radio
   subpíxel en el bottleneck → la información se diluye en BatchNorm y se
   pierde irrecuperablemente.

2. **Skip connections.** Las skip de niveles altos (poco downsampleadas)
   sí preservan detalle fino, pero el decoder debe combinarlas con
   features semánticas profundas. En los Attention Gates, la señal de
   gating del decoder es de baja resolución, por lo que su atención
   tiende a privilegiar estructuras grandes — penalizando capilares.

3. **Desbalance intra-clase.** Aunque "vaso" es una sola clase, los
   capilares aportan muchos menos píxeles que las arterias gruesas.
   La Dice loss, agregada sobre todos los píxeles-vaso, está dominada
   por los gruesos, así que el gradiente "premia" más segmentar bien una
   arteria que un capilar.

4. **Anti-aliasing en resize.** Al redimensionar a {img_size}×{img_size}
   con interpolación bilinear, los capilares de 1-px del original quedan
   sub-resolvibles. La máscara GT se redimensiona con NEAREST (preserva
   binariedad) pero pierde continuidad.

## Mitigaciones posibles (a discutir en el informe)

- Reducir el número de downsamplings (depth=3 en lugar de 4).
- Entrenar en parches de alta resolución sin resize (e.g. 96×96 sin
  reescalar).
- Añadir una loss específica de capilares (Dice solo sobre la categoría
  "fino").
- Pretraining con tareas de detección de bordes (Sobel, Canny) para
  sesgar el modelo hacia frecuencias altas.
"""
    with open(interp_path, "w") as f:
        f.write(interp)
    print(f"  ✓ {interp_path}")

    print(f"\n✓ Análisis completo guardado en: {out_dir}")


if __name__ == "__main__":
    main()
