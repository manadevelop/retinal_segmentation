"""
utils/trainer.py — Bucle de entrenamiento para segmentación retinal.

Gestiona:
  - Entrenamiento epoch-by-epoch con barra tqdm
  - Validación con early stopping por F1 (Dice)
  - Guardado del mejor modelo
  - Registro de métricas en JSON
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import compute_metrics_batch, print_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer,
        scheduler,
        device: str,
        output_dir: Path,
        cfg: Dict[str, Any],
        logger,
    ):
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.device     = device
        self.output_dir = Path(output_dir)
        self.cfg        = cfg
        self.logger     = logger

        self.best_f1          = 0.0
        self.patience_counter = 0
        self.patience         = cfg.get("early_stopping_patience", 20)

        self.history: Dict = {
            "train_loss": [], "val_loss": [],
            "val_f1": [], "val_sensitivity": [],
            "val_specificity": [], "val_auc": [],
        }

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="  Train", leave=False, ncols=80):
            imgs, masks = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        accum = {k: [] for k in
                 ("sensibilidad", "especificidad", "precision",
                  "f1", "iou", "accuracy", "auc_roc")}

        for batch in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
            imgs, masks = batch[0].to(self.device), batch[1].to(self.device)
            logits      = self.model(imgs)
            loss        = self.criterion(logits, masks)
            total_loss += loss.item()

            proba   = torch.sigmoid(logits)
            # Usar FOV si el dataset lo proporciona para no evaluar fuera de la retina
            fov = batch[2].to(self.device) if len(batch) > 2 else None
            metrics = compute_metrics_batch(proba, masks, valid_mask=fov)
            for k in accum:
                accum[k].append(metrics[k])

        result = {"loss": total_loss / max(len(loader), 1)}
        for k, vals in accum.items():
            # nanmean para AUC que puede ser NaN si batch sin positivos
            clean_vals = [v for v in vals if not np.isnan(v)]
            result[k] = float(np.mean(clean_vals)) if clean_vals else float("nan")
        return result

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        self.logger.info(f"Iniciando entrenamiento por {epochs} épocas...")
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            train_loss  = self._train_epoch(train_loader)
            val_metrics = self._val_epoch(val_loader)
            self.scheduler.step()

            val_f1   = val_metrics["f1"]
            val_sens = val_metrics["sensibilidad"]
            val_spec = val_metrics["especificidad"]
            val_auc  = val_metrics["auc_roc"]
            val_loss = val_metrics["loss"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)
            self.history["val_sensitivity"].append(val_sens)
            self.history["val_specificity"].append(val_spec)
            self.history["val_auc"].append(val_auc)

            self.logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"TrainLoss={train_loss:.4f} | "
                f"ValLoss={val_loss:.4f} | "
                f"F1={val_f1:.4f} | "
                f"Sens={val_sens:.4f} | "
                f"Spec={val_spec:.4f} | "
                f"AUC={val_auc:.4f}"
            )

            if val_f1 > self.best_f1 or not (self.output_dir / "best_model.pt").exists():
                self.best_f1          = val_f1
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt")
                self.logger.info(f"  ✓ Nuevo mejor modelo (F1={val_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(
                        f"Early stopping en época {epoch} "
                        f"(sin mejora en {self.patience} épocas)."
                    )
                    break

        elapsed = (time.time() - t0) / 60
        self.logger.info(f"Entrenamiento completado en {elapsed:.1f} min.")
        self._save_history()

    def evaluate(self, loader: DataLoader, split: str = "test") -> Dict:
        ckpt = self.output_dir / "best_model.pt"
        if ckpt.exists():
            self.model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True)
            )
            self.logger.info(f"Cargado: {ckpt}")

        metrics = self._val_epoch(loader)

        # Guardar JSON ANTES de imprimir (robusto a errores de formato)
        out_path = self.output_dir / f"{split}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Métricas guardadas en {out_path}")

        self.logger.info(f"\n=== Resultados en {split.upper()} ===")
        print_metrics(metrics, prefix=split)
        return metrics

    def _save_checkpoint(self, name: str = "best_model.pt"):
        torch.save(self.model.state_dict(), self.output_dir / name)

    def _save_history(self):
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
