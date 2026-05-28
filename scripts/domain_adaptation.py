"""
scripts/domain_adaptation.py — Experimento de generalización y adaptación de dominio.

Ejecuta:
  1. Evaluación directa: modelo entrenado en DRIVE → evaluado en CHASE_DB1
     (sin adaptación) → mide la brecha de rendimiento
  2. Fine-tuning: ajuste fino en 20% de CHASE_DB1 → evalúa mejora
  3. CLAHE: evalúa el efecto del preprocesamiento CLAHE en la brecha

Uso:
  python scripts/domain_adaptation.py \
    --config configs/train_attention_unet.yaml \
    --checkpoint outputs/attention_unet_drive/best_model.pt \
    --chase_root data/chase_db1 \
    --out_dir results/domain_adaptation
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.attention_unet import AttentionUNet
from models.unet import UNet
from data.dataset import ChaseDB1Dataset
from data.transforms import get_transforms, ValTransform
from utils.losses import get_loss
from utils.metrics import compute_metrics_batch, print_metrics
from utils.trainer import Trainer
from utils.logger import setup_logger


def load_model(cfg: dict, checkpoint: str, device: str):
    model_name = cfg["model"]["name"]
    if model_name == "attention_unet":
        model = AttentionUNet(
            in_channels=cfg["model"].get("in_channels", 3),
            out_channels=cfg["model"].get("out_channels", 1),
            base_channels=cfg["model"].get("base_channels", 64),
            depth=cfg["model"].get("depth", 4),
        )
    else:
        model = UNet(
            in_channels=cfg["model"].get("in_channels", 3),
            out_channels=cfg["model"].get("out_channels", 1),
            base_channels=cfg["model"].get("base_channels", 64),
            depth=cfg["model"].get("depth", 4),
        )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def evaluate_loader(model, loader, device, criterion=None):
    model.eval()
    sens_l, spec_l, f1_l, auc_l, loss_l = [], [], [], [], []

    for batch in loader:
        imgs, masks = batch[0].to(device), batch[1].to(device)
        fov = batch[2].to(device) if len(batch) > 2 else None
        logits      = model(imgs)
        proba       = torch.sigmoid(logits)

        if criterion:
            loss_l.append(criterion(logits, masks).item())

        m = compute_metrics_batch(proba, masks, valid_mask=fov)
        sens_l.append(m["sensibilidad"])
        spec_l.append(m["especificidad"])
        f1_l.append(m["f1"])
        auc_l.append(m["auc_roc"])

    return {
        "sensibilidad":  float(np.mean(sens_l)),
        "especificidad": float(np.mean(spec_l)),
        "f1":            float(np.mean(f1_l)),
        "auc_roc":       float(np.mean([v for v in auc_l if not np.isnan(v)])) if any(not np.isnan(v) for v in auc_l) else float("nan"),
        "loss":          float(np.mean(loss_l)) if loss_l else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--chase_root", type=str,
                        default="data/chase_db1")
    parser.add_argument("--out_dir",    type=str,
                        default="results/domain_adaptation")
    parser.add_argument("--finetune_ratio", type=float, default=0.2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger  = setup_logger("domain_adaptation")

    img_size = cfg["data"].get("img_size", 512)
    _, val_tf = get_transforms(img_size=img_size, strategy="minimal")
    val_tf_clahe = ValTransform(img_size=img_size, use_clahe=True)

    results = {}

    # ── EXPERIMENTO 1: Evaluación directa sin adaptación ──────────────────
    logger.info("="*60)
    logger.info("EXPERIMENTO 1: DRIVE → CHASE_DB1 (sin adaptación)")
    logger.info("="*60)

    model = load_model(cfg, args.checkpoint, device)
    criterion = get_loss(cfg["training"]["loss"], cfg["training"])

    chase_test = ChaseDB1Dataset(
        args.chase_root, split="all", transform=val_tf
    )
    chase_loader = DataLoader(
        chase_test, batch_size=2, shuffle=False, num_workers=2
    )

    metrics_no_adapt = evaluate_loader(model, chase_loader, device, criterion)
    logger.info("Sin adaptación:")
    print_metrics(metrics_no_adapt, prefix="CHASE_DB1 directo")
    results["sin_adaptacion"] = metrics_no_adapt

    # ── EXPERIMENTO 2: Efecto de CLAHE ────────────────────────────────────
    logger.info("="*60)
    logger.info("EXPERIMENTO 2: CLAHE como preprocesamiento")
    logger.info("="*60)

    chase_clahe = ChaseDB1Dataset(
        args.chase_root, split="all", transform=val_tf_clahe
    )
    chase_clahe_loader = DataLoader(
        chase_clahe, batch_size=2, shuffle=False, num_workers=2
    )

    metrics_clahe = evaluate_loader(
        model, chase_clahe_loader, device, criterion
    )
    logger.info("Con CLAHE:")
    print_metrics(metrics_clahe, prefix="CHASE_DB1 + CLAHE")
    results["con_clahe"] = metrics_clahe

    mejora_clahe = metrics_clahe["f1"] - metrics_no_adapt["f1"]
    logger.info(f"Mejora CLAHE sobre F1: {mejora_clahe:+.4f}")

    # ── EXPERIMENTO 3: Fine-tuning en subconjunto CHASE_DB1 ───────────────
    logger.info("="*60)
    logger.info(f"EXPERIMENTO 3: Fine-tuning en {args.finetune_ratio*100:.0f}% de CHASE_DB1")
    logger.info("="*60)

    train_tf, val_tf2 = get_transforms(
        img_size=img_size,
        strategy="standard",
        use_clahe=True,
    )

    full_chase = ChaseDB1Dataset(
        args.chase_root, split="all", transform=train_tf
    )
    n_ft   = max(1, int(len(full_chase) * args.finetune_ratio))
    n_eval = len(full_chase) - n_ft

    ft_ds, eval_ds = random_split(
        full_chase, [n_ft, n_eval],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(
        f"Fine-tuning con {n_ft} imágenes, "
        f"evaluando en {n_eval} imágenes"
    )

    ft_loader   = DataLoader(ft_ds,   batch_size=2, shuffle=True,  num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, num_workers=2)

    # Recargar modelo original y hacer fine-tuning con LR muy baja
    model_ft   = load_model(cfg, args.checkpoint, device)
    optimizer  = optim.AdamW(
        model_ft.parameters(),
        lr=cfg["training"]["lr"] * 0.1,
        weight_decay=1e-4,
    )
    scheduler  = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)
    ft_out_dir = out_dir / "finetuned_model"
    ft_out_dir.mkdir(exist_ok=True)

    trainer = Trainer(
        model=model_ft,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=ft_out_dir,
        cfg={**cfg, "early_stopping_patience": 10},
        logger=logger,
    )
    trainer.fit(ft_loader, eval_loader, epochs=20)

    # ── Evaluar SÓLO en el subconjunto held-out (no en finetune) ──────────
    # IMPORTANTE: evaluamos en `eval_ds` para evitar fuga de datos
    # (antes el código evaluaba en split="all" que incluía las imágenes
    #  de fine-tuning → métricas infladas).
    #
    # Recreamos eval_ds con transform de validación (sin augmentation,
    # con CLAHE para que la evaluación sea consistente).
    full_chase_val = ChaseDB1Dataset(
        args.chase_root, split="all", transform=val_tf2
    )
    # Mismo random_split (misma semilla) para obtener los mismos índices
    _, eval_ds_val = random_split(
        full_chase_val, [n_ft, n_eval],
        generator=torch.Generator().manual_seed(42)
    )
    chase_eval_loader = DataLoader(
        eval_ds_val, batch_size=2, shuffle=False, num_workers=2
    )

    metrics_ft = evaluate_loader(
        model_ft, chase_eval_loader, device, criterion
    )
    logger.info(f"Con fine-tuning (evaluado en {n_eval} imágenes held-out):")
    print_metrics(metrics_ft, prefix="CHASE_DB1 + fine-tuning")
    results["con_finetune"] = metrics_ft
    results["finetune_metadata"] = {
        "n_finetune": n_ft,
        "n_eval_held_out": n_eval,
        "note": "Evaluación realizada solo en el subconjunto NO usado para fine-tuning",
    }

    # ── RESUMEN COMPARATIVO ────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("RESUMEN — Brecha de dominio DRIVE → CHASE_DB1")
    logger.info("="*60)

    metricas_clave = ["sensibilidad", "especificidad", "f1", "auc_roc"]

    print(f"\n  {'Configuración':<25} "
          + "  ".join(f"{m:>12}" for m in metricas_clave))
    print("  " + "-"*75)

    for nombre, res in [
        ("Sin adaptación",  results["sin_adaptacion"]),
        ("+ CLAHE",         results["con_clahe"]),
        ("+ Fine-tuning",   results["con_finetune"]),
    ]:
        vals = "  ".join(f"{res[m]:>12.4f}" for m in metricas_clave)
        print(f"  {nombre:<25} {vals}")

    brecha_f1 = results["sin_adaptacion"]["f1"] - results["con_finetune"]["f1"]
    logger.info(
        f"\nReducción de brecha F1 tras fine-tuning: {abs(brecha_f1):.4f}"
    )

    # Guardar resultados
    with open(out_dir / "domain_adaptation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()
