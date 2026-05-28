"""
train.py — Entrenamiento de modelos para segmentación de vasos retinianos.

Uso:
  python src/train.py --config configs/train_attention_unet.yaml
  python src/train.py --config configs/train_unet.yaml
  python src/train.py --config configs/ablation_dice.yaml
"""

import argparse
import os
import random
import sys
from pathlib import Path

# Asegurar que src/ esté en sys.path para que las importaciones
# `from models...`, `from data...`, `from utils...` funcionen
# independientemente de cómo se invoque este script.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import yaml

# wandb opcional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from models.attention_unet import AttentionUNet
from models.unet import UNet
from data.dataset import DriveDataset, StareDataset, ChaseDB1Dataset
from data.transforms import get_transforms
from utils.losses import get_loss
from utils.trainer import Trainer
from utils.logger import setup_logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_model(cfg: dict) -> torch.nn.Module:
    model_name    = cfg["model"]["name"]
    in_channels   = cfg["model"].get("in_channels",   3)
    out_channels  = cfg["model"].get("out_channels",  1)
    base_channels = cfg["model"].get("base_channels", 64)
    depth         = cfg["model"].get("depth",         4)
    bilinear      = cfg["model"].get("bilinear",      True)
    dropout_rate  = cfg["model"].get("dropout_rate",  0.1)

    if model_name == "attention_unet":
        return AttentionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            bilinear=bilinear,
            dropout_rate=dropout_rate,
        )
    elif model_name == "unet":
        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            bilinear=bilinear,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


def build_datasets(cfg: dict):
    data_cfg   = cfg["data"]
    img_size   = data_cfg.get("img_size",   512)
    strategy   = data_cfg.get("augmentation", "standard")
    use_clahe  = data_cfg.get("use_clahe",  False)
    dataset    = data_cfg.get("dataset",    "drive")

    train_tf, val_tf = get_transforms(
        img_size=img_size,
        strategy=strategy,
        use_clahe=use_clahe,
    )

    if dataset == "drive":
        root = data_cfg.get("drive_root", "data/drive")

        # DRIVE oficial: training tiene anotaciones públicas; test oficial no siempre
        # incluye 1st_manual. Si test/1st_manual existe, se usa el split fijo. Si no,
        # se crea un holdout interno reproducible desde training para obtener métricas
        # válidas con ground truth.
        test_manual = Path(root) / "test" / "1st_manual"
        has_test_gt = test_manual.exists() and any(test_manual.glob("*"))

        if has_test_gt:
            train_ds = DriveDataset(root, split="train", transform=train_tf, use_clahe=use_clahe)
            val_ds = DriveDataset(root, split="test", transform=val_tf, use_clahe=use_clahe)
            test_ds = val_ds
        else:
            from torch.utils.data import Subset
            full_train_tf = DriveDataset(root, split="train", transform=train_tf, use_clahe=use_clahe)
            full_val_tf = DriveDataset(root, split="train", transform=val_tf, use_clahe=use_clahe)
            n = len(full_train_tf)
            idx = list(range(n))
            rng = np.random.default_rng(cfg.get("seed", 42))
            rng.shuffle(idx)
            n_train = max(1, int(round(0.70 * n)))
            n_val = max(1, int(round(0.15 * n)))
            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:] or val_idx
            train_ds = Subset(full_train_tf, train_idx)
            val_ds = Subset(full_val_tf, val_idx)
            test_ds = Subset(full_val_tf, test_idx)

    elif dataset == "chase_db1":
        root       = data_cfg.get("chase_root", "data/chase_db1")
        full_ds    = ChaseDB1Dataset(root, split="all",
                                     transform=train_tf,
                                     use_clahe=use_clahe)
        n          = len(full_ds)
        n_train    = int(n * 0.7)
        n_val      = n - n_train
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        test_ds    = ChaseDB1Dataset(root, split="test",
                                     transform=val_tf,
                                     use_clahe=use_clahe)
    else:
        raise ValueError(f"Dataset desconocido: {dataset}")

    return train_ds, val_ds, test_ds


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    logger = setup_logger(cfg["experiment_name"])

    device        = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    logger.info(f"Dispositivo: {device}")

    # ── Datasets ───────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = build_datasets(cfg)
    logger.info(
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    num_workers = cfg["training"].get("num_workers", 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"].get("val_batch_size", 2),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"].get("val_batch_size", 2),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── Modelo ─────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parámetros totales: {n_params:,}")

    # ── Pérdida ────────────────────────────────────────────────────────────
    criterion = get_loss(cfg["training"]["loss"], cfg["training"])

    # ── Optimizador ────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("lr_min", 1e-6),
    )

    # ── W&B opcional ───────────────────────────────────────────────────────
    use_wandb = cfg.get("use_wandb", False) and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="retinal-segmentation",
            name=cfg["experiment_name"],
            config=cfg,
        )

    # ── Entrenamiento ──────────────────────────────────────────────────────
    output_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        cfg=cfg,
        logger=logger,
    )

    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])

    logger.info("Evaluando en conjunto de prueba...")
    trainer.evaluate(test_loader, split="test")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
