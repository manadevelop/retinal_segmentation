"""
src/data/dataset.py — Datasets para segmentación de vasos retinianos.

Soporta DRIVE, STARE y CHASE_DB1 con las estructuras observadas en Google Drive
y con nombres originales de los datasets. No crea máscaras vacías silenciosamente:
si falta una anotación requerida, lanza un error explícito.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from data.transforms import to_tensor_normalized, to_tensor_mask

IMG_EXTS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.ppm")


def _files(directory: Path, patterns=IMG_EXTS) -> List[Path]:
    out: List[Path] = []
    if directory.exists():
        for pat in patterns:
            out.extend(directory.glob(pat))
    return sorted([p for p in out if p.is_file() and not p.name.startswith("._")])


def _open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _open_l(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def _ones_fov_like(img: Image.Image) -> Image.Image:
    return Image.fromarray(np.ones(np.array(img).shape[:2], dtype=np.uint8) * 255)


class DriveDataset(Dataset):
    """DRIVE con split fijo `training`/`test`.

    Estructura esperada:
      root/training/images, root/training/1st_manual, root/training/mask
      root/test/images,     root/test/1st_manual,     root/test/mask

    Nota: en la fuente oficial de DRIVE, las anotaciones test no son públicas; si
    `test/1st_manual` no existe, esta clase falla de forma explícita. Para métricas
    locales se recomienda usar un holdout desde `training` (ver train.py).
    """

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None, use_clahe: bool = False):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_clahe = use_clahe
        split_dir = "training" if split in ("train", "training") else "test"
        self.img_dir = self.root / split_dir / "images"
        self.mask_dir = self.root / split_dir / "1st_manual"
        self.fov_dir = self.root / split_dir / "mask"
        self.images = _files(self.img_dir, ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"))
        if not self.images:
            raise RuntimeError(f"No se encontraron imágenes DRIVE en {self.img_dir}.")
        if not self.mask_dir.exists() or not _files(self.mask_dir, ("*.gif", "*.png", "*.tif", "*.tiff")):
            raise RuntimeError(
                f"No se encontraron máscaras manuales DRIVE en {self.mask_dir}. "
                "La carpeta `mask/` es FOV, no ground truth de vasos."
            )

    def __len__(self) -> int:
        return len(self.images)

    def _id(self, img_path: Path) -> str:
        return img_path.stem.split("_")[0]

    def _find_mask(self, img_path: Path) -> Path:
        sid = self._id(img_path)
        candidates = []
        for pat in (f"{sid}_manual1.*", f"{sid}*manual*", f"{sid}*.gif", f"{sid}*.png", f"{sid}*.tif"):
            candidates.extend(self.mask_dir.glob(pat))
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            raise RuntimeError(f"No se encontró máscara manual para {img_path.name} en {self.mask_dir}.")
        return sorted(candidates)[0]

    def _find_fov(self, img_path: Path, img: Image.Image) -> Image.Image:
        sid = self._id(img_path)
        candidates = []
        if self.fov_dir.exists():
            for pat in (f"{sid}_training_mask.*", f"{sid}_test_mask.*", f"{sid}*mask*", f"{sid}.*"):
                candidates.extend(self.fov_dir.glob(pat))
        candidates = [p for p in candidates if p.is_file()]
        return _open_l(sorted(candidates)[0]) if candidates else _ones_fov_like(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        img = _open_rgb(img_path)
        mask = _open_l(self._find_mask(img_path))
        fov = self._find_fov(img_path, img)
        if self.transform:
            return self.transform(img, mask, fov)
        return to_tensor_normalized(img), to_tensor_mask(mask), to_tensor_mask(fov)


class StareDataset(Dataset):
    """STARE, 20 imágenes y anotaciones de Hoover (`ah`) o Kouznetsova (`vk`).

    Soporta:
      data/stare/images/im0001.ppm o im0001.ppm.png
      data/stare/masks/im0001.ah.ppm o im0001.ah.ppm.png
      data/stare/labels-ah/... y data/stare/labels-vk/...
    """

    def __init__(self, root: str, annotator: str = "ah", transform: Optional[Callable] = None, use_clahe: bool = False):
        self.root = Path(root)
        self.annotator = annotator
        self.transform = transform
        self.use_clahe = use_clahe
        self.img_dir = self.root / "images"
        if (self.root / f"labels-{annotator}").exists():
            self.mask_dir = self.root / f"labels-{annotator}"
        elif (self.root / "masks").exists():
            self.mask_dir = self.root / "masks"
        else:
            raise RuntimeError(f"No se encontró carpeta de máscaras STARE para anotador {annotator} en {root}.")

        all_imgs = _files(self.img_dir, ("*.ppm", "*.ppm.png", "*.png", "*.jpg", "*.jpeg"))
        # Evitar duplicados cuando existen im0001.ppm e im0001.ppm.png.
        by_id = {}
        for p in all_imgs:
            sid = p.name.split(".")[0]
            if sid not in by_id or p.suffix.lower() == ".ppm":
                by_id[sid] = p
        self.images = [by_id[k] for k in sorted(by_id)]
        if not self.images:
            raise RuntimeError(f"No se encontraron imágenes STARE en {self.img_dir}.")

    def __len__(self) -> int:
        return len(self.images)

    def _id(self, img_path: Path) -> str:
        return img_path.name.split(".")[0]

    def _find_mask(self, img_path: Path) -> Path:
        sid = self._id(img_path)
        candidates = []
        for pat in (f"{sid}.{self.annotator}.ppm", f"{sid}.{self.annotator}.ppm.png", f"{sid}.{self.annotator}.*", f"{sid}*{self.annotator}*"):
            candidates.extend(self.mask_dir.glob(pat))
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            raise RuntimeError(f"No se encontró máscara STARE {self.annotator} para {img_path.name} en {self.mask_dir}.")
        return sorted(candidates)[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        img = _open_rgb(img_path)
        mask = _open_l(self._find_mask(img_path))
        fov = _ones_fov_like(img)
        if self.transform:
            return self.transform(img, mask, fov)
        return to_tensor_normalized(img), to_tensor_mask(mask), to_tensor_mask(fov)


class ChaseDB1Dataset(Dataset):
    """CHASE_DB1 para evaluación cross-domain.

    Soporta tu estructura:
      images/01_test.tif -> labels/01_manual1.tif -> mask/01.tif
    y variantes originales tipo Image_11R.* -> Image_11R_1stHO.*.
    """

    def __init__(self, root: str, split: str = "all", transform: Optional[Callable] = None, use_clahe: bool = False):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_clahe = use_clahe
        self.img_dir = self.root / "images"
        if (self.root / "labels").exists():
            self.mask_dir = self.root / "labels"
        elif (self.root / "1st_manual").exists():
            self.mask_dir = self.root / "1st_manual"
        else:
            raise RuntimeError(f"No se encontró labels/ o 1st_manual/ en {root}.")
        self.fov_dir = self.root / "mask" if (self.root / "mask").exists() else None

        all_images = _files(self.img_dir)
        if not all_images:
            raise RuntimeError(f"No se encontraron imágenes CHASE_DB1 en {self.img_dir}.")

        if split == "all":
            self.images = all_images
        else:
            n = len(all_images)
            n_train = max(1, int(round(0.2 * n)))
            self.images = all_images[:n_train] if split == "train" else all_images[n_train:]
            if not self.images:
                self.images = all_images

    def __len__(self) -> int:
        return len(self.images)

    def _id(self, img_path: Path) -> str:
        stem = img_path.stem
        return stem.replace("_test", "") if stem.endswith("_test") else stem

    def _find_mask(self, img_path: Path) -> Path:
        sid, stem = self._id(img_path), img_path.stem
        candidates = []
        for pat in (f"{sid}_manual1.*", f"{sid}_manual.*", f"{sid}*manual*", f"{stem}_1stHO.*", f"{stem}*1stHO*", f"{stem}*"):
            candidates.extend(self.mask_dir.glob(pat))
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            raise RuntimeError(f"No se encontró máscara CHASE_DB1 para {img_path.name} en {self.mask_dir}.")
        return sorted(candidates)[0]

    def _find_fov(self, img_path: Path, img: Image.Image) -> Image.Image:
        if self.fov_dir is None:
            return _ones_fov_like(img)
        sid, stem = self._id(img_path), img_path.stem
        candidates = []
        for pat in (f"{sid}.*", f"{stem}.*", f"{sid}*mask*", f"{stem}*mask*"):
            candidates.extend(self.fov_dir.glob(pat))
        candidates = [p for p in candidates if p.is_file()]
        return _open_l(sorted(candidates)[0]) if candidates else _ones_fov_like(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        img = _open_rgb(img_path)
        mask = _open_l(self._find_mask(img_path))
        fov = self._find_fov(img_path, img)
        if self.transform:
            return self.transform(img, mask, fov)
        return to_tensor_normalized(img), to_tensor_mask(mask), to_tensor_mask(fov)
