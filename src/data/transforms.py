"""
src/data/transforms.py — Transformaciones sincronizadas para segmentación retinal.

Soporta imagen, máscara de vasos y máscara FOV. La binarización de máscaras es
robusta para formatos 0/1, 0/255, 0/65535 y PPM/PNG/TIF.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def apply_clahe_pil(img: Image.Image) -> Image.Image:
    """Aplica CLAHE al canal verde de una imagen RGB retinal."""
    img_np = np.array(img.convert("RGB")).copy()
    green = img_np[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_np[:, :, 1] = clahe.apply(green)
    return Image.fromarray(img_np)


def to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    """Convierte imagen PIL RGB a tensor normalizado con estadísticas ImageNet."""
    img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    tensor = torch.from_numpy(img_np)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def to_tensor_mask(mask: Image.Image) -> torch.Tensor:
    """Convierte una máscara PIL a tensor binario {0,1} con shape (1,H,W).

    No usa un umbral fijo >127 porque algunos datasets guardan anotaciones como
    0/1. La regla robusta es: todo píxel no-cero es foreground. Esto funciona
    para máscaras de vasos y para FOV masks binarias.
    """
    arr = np.array(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    mask_bin = (arr > 0).astype(np.float32)
    return torch.from_numpy(mask_bin[np.newaxis])


def elastic_transform(
    img: Image.Image,
    mask: Image.Image,
    fov: Optional[Image.Image] = None,
    alpha: float = 35,
    sigma: float = 5,
) -> Tuple[Image.Image, Image.Image, Optional[Image.Image]]:
    """Deformación elástica sincronizada para imagen, máscara y FOV."""
    img_np = np.array(img.convert("RGB"))
    mask_np = np.array(mask)
    fov_np = np.array(fov) if fov is not None else None
    h, w = img_np.shape[:2]

    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    img_t = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask_t = cv2.remap(mask_np, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fov_t = None
    if fov_np is not None:
        fov_t = cv2.remap(fov_np, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return Image.fromarray(img_t), Image.fromarray(mask_t), (Image.fromarray(fov_t) if fov_t is not None else None)


class RetinalTransform:
    """Transformaciones sincronizadas para entrenamiento."""

    def __init__(self, img_size: int = 512, strategy: str = "standard", use_clahe: bool = False):
        self.img_size = img_size
        self.strategy = strategy
        self.use_clahe = use_clahe

    def __call__(self, img: Image.Image, mask: Image.Image, fov: Optional[Image.Image] = None):
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        if fov is not None:
            fov = fov.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        if self.strategy in ("standard", "aggressive"):
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT); mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                if fov is not None: fov = fov.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM); mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                if fov is not None: fov = fov.transpose(Image.FLIP_TOP_BOTTOM)
            angle = random.uniform(-20, 20)
            img = img.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
            if fov is not None:
                fov = fov.rotate(angle, resample=Image.NEAREST)

            if random.random() > 0.3:
                from PIL import ImageEnhance
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
                gamma = random.uniform(0.85, 1.15)
                arr = np.array(img).astype(np.float32) / 255.0
                arr = np.clip(arr ** gamma, 0, 1)
                img = Image.fromarray((arr * 255).astype(np.uint8))

        if self.strategy == "aggressive" and random.random() > 0.5:
            img, mask, fov = elastic_transform(img, mask, fov)

        if self.use_clahe:
            img = apply_clahe_pil(img)

        img_t = to_tensor_normalized(img)
        mask_t = to_tensor_mask(mask)
        if fov is not None:
            fov_t = to_tensor_mask(fov)
            return img_t, mask_t, fov_t
        return img_t, mask_t


class ValTransform:
    """Transformaciones deterministas para validación/test."""

    def __init__(self, img_size: int = 512, use_clahe: bool = False):
        self.img_size = img_size
        self.use_clahe = use_clahe

    def __call__(self, img: Image.Image, mask: Image.Image, fov: Optional[Image.Image] = None):
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        if fov is not None:
            fov = fov.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        if self.use_clahe:
            img = apply_clahe_pil(img)

        img_t = to_tensor_normalized(img)
        mask_t = to_tensor_mask(mask)
        if fov is not None:
            return img_t, mask_t, to_tensor_mask(fov)
        return img_t, mask_t


def get_transforms(img_size: int = 512, strategy: str = "standard", use_clahe: bool = False):
    return (
        RetinalTransform(img_size=img_size, strategy=strategy, use_clahe=use_clahe),
        ValTransform(img_size=img_size, use_clahe=use_clahe),
    )
