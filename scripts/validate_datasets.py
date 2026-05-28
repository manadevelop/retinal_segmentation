"""Validación rápida de estructura y máscaras antes de entrenar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.dataset import DriveDataset, ChaseDB1Dataset, StareDataset
from data.transforms import ValTransform


def inspect_dataset(name, builder, required=True):
    try:
        ds = builder()
        if len(ds) == 0:
            raise RuntimeError("dataset vacío")
        x, y, fov = ds[0]
        mask_sum = float(y.sum())
        fov_sum = float(fov.sum())
        if mask_sum <= 0:
            raise RuntimeError("la primera máscara de vasos queda vacía después de binarizar")
        if fov_sum <= 0:
            raise RuntimeError("la máscara FOV queda vacía")
        print(f"✓ {name:<18} n={len(ds):<3} image={tuple(x.shape)} mask_sum={mask_sum:.1f} fov_sum={fov_sum:.1f}")
        return True
    except Exception as e:
        if required:
            raise RuntimeError(f"Falla validación {name}: {e}") from e
        print(f"⚠ {name:<18} omitido/no válido: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive_root", default="data/drive")
    parser.add_argument("--chase_root", default="data/chase_db1")
    parser.add_argument("--stare_root", default="data/stare")
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()

    tf = ValTransform(img_size=args.img_size)
    print("Validando datasets y máscaras...")
    inspect_dataset("DRIVE train", lambda: DriveDataset(args.drive_root, split="train", transform=tf), required=True)
    inspect_dataset("DRIVE test", lambda: DriveDataset(args.drive_root, split="test", transform=tf), required=False)
    inspect_dataset("CHASE_DB1", lambda: ChaseDB1Dataset(args.chase_root, split="all", transform=tf), required=True)
    inspect_dataset("STARE AH", lambda: StareDataset(args.stare_root, annotator="ah", transform=tf), required=False)
    inspect_dataset("STARE VK", lambda: StareDataset(args.stare_root, annotator="vk", transform=tf), required=False)
    print("Validación finalizada. Si DRIVE test fue omitido, el entrenamiento usará holdout de DRIVE/training para métricas locales.")


if __name__ == "__main__":
    main()
