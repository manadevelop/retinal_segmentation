"""
scripts/visualize_attention.py

Visualiza mapas de atención de Attention U-Net.

Este script genera figuras cualitativas con:
- imagen de fundus,
- ground truth de vasos,
- predicción binaria,
- mapas de atención internos del decoder.

Importante:
DRIVE test oficial normalmente no incluye máscaras manuales públicas.
Si data/drive/test/1st_manual no existe, el script usa automáticamente
data/drive/training para evitar evaluar/visualizar contra máscaras FOV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.dataset import DriveDataset
from data.transforms import ValTransform
from models.attention_unet import AttentionUNet


def has_drive_manual_masks(drive_root: str | Path, split: str) -> bool:
    """
    Verifica si DRIVE tiene máscaras manuales reales para el split solicitado.

    La carpeta mask/ de DRIVE es FOV, no ground truth de vasos.
    """
    drive_root = Path(drive_root)
    split_dir = "training" if split == "train" else "test"
    manual_dir = drive_root / split_dir / "1st_manual"

    if not manual_dir.exists():
        return False

    patterns = ["*.gif", "*.png", "*.tif", "*.tiff", "*.ppm"]
    files = []
    for pattern in patterns:
        files.extend(list(manual_dir.glob(pattern)))

    return len(files) > 0


def resolve_drive_split(drive_root: str | Path, requested_split: str) -> str:
    """
    Decide qué split usar para visualización.

    - Si requested_split es train, usa train.
    - Si requested_split es test y existe test/1st_manual, usa test.
    - Si requested_split es test pero no existe ground truth, usa train.
    """
    drive_root = Path(drive_root)

    if requested_split == "train":
        if not has_drive_manual_masks(drive_root, "train"):
            raise RuntimeError(
                f"No se encontraron máscaras manuales en "
                f"{drive_root / 'training' / '1st_manual'}."
            )
        return "train"

    if requested_split != "test":
        raise ValueError("requested_split debe ser 'train' o 'test'.")

    if has_drive_manual_masks(drive_root, "test"):
        return "test"

    print(
        "⚠ DRIVE test no tiene máscaras manuales en test/1st_manual. "
        "La carpeta test/mask es FOV, no ground truth de vasos. "
        "Se usará DRIVE/training para visualización cualitativa."
    )

    if not has_drive_manual_masks(drive_root, "train"):
        raise RuntimeError(
            f"No se encontraron máscaras manuales en "
            f"{drive_root / 'training' / '1st_manual'}."
        )

    return "train"


def load_checkpoint_safely(checkpoint_path: str | Path, device: str):
    """
    Carga checkpoints de forma compatible con distintas versiones de PyTorch.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


def register_attention_hooks(model: AttentionUNet):
    """
    Registra forward hooks sobre los módulos psi de cada Attention Gate.
    """
    captures: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            captures[name] = output.detach().cpu()

        return hook

    for i, decoder in enumerate(model.decoders):
        if hasattr(decoder, "att") and hasattr(decoder.att, "psi"):
            handle = decoder.att.psi.register_forward_hook(
                make_hook(f"level_{i}")
            )
            handles.append(handle)

    if len(handles) == 0:
        print(
            "⚠ No se encontraron Attention Gates con atributo decoder.att.psi. "
            "Se generarán imágenes sin mapas de atención internos."
        )

    return captures, handles


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Desnormaliza imagen normalizada con estadísticas ImageNet.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = tensor.detach().cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()

    return np.clip(img, 0.0, 1.0)


def unpack_batch(batch):
    """
    Soporta datasets que devuelven:
    - image, mask
    - image, mask, fov
    """
    if len(batch) == 2:
        image, mask = batch
        fov = None
    elif len(batch) == 3:
        image, mask, fov = batch
    else:
        raise RuntimeError(
            f"Formato de batch no soportado. Se esperaban 2 o 3 elementos, "
            f"pero llegó {len(batch)}."
        )

    return image, mask, fov


def resize_attention_map(attention: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    """
    Redimensiona mapa de atención al tamaño de la imagen.
    """
    if attention.ndim == 4:
        attention = attention[:1]

    attention = F.interpolate(
        attention.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    attention_np = attention.squeeze().numpy()

    if attention_np.ndim > 2:
        attention_np = attention_np[0]

    return attention_np


def plot_attention_sample(
    image_np: np.ndarray,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    captures: Dict[str, torch.Tensor],
    out_path: Path,
    title: str,
):
    """
    Genera figura con imagen, ground truth, predicción y mapas de atención.
    """
    attention_items = sorted(captures.items(), key=lambda x: x[0])
    n_attention = len(attention_items)

    n_cols = max(3, n_attention)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(4 * n_cols, 8),
    )

    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Imagen", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_np, cmap="gray")
    axes[0, 1].set_title("Ground truth", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_np, cmap="gray")
    axes[0, 2].set_title("Predicción", fontsize=11)
    axes[0, 2].axis("off")

    for col in range(3, n_cols):
        axes[0, col].axis("off")

    h, w = gt_np.shape

    for col, (name, attention) in enumerate(attention_items):
        if col >= n_cols:
            break

        attention_np = resize_attention_map(attention, size=(h, w))

        axes[1, col].imshow(image_np, alpha=0.55)
        axes[1, col].imshow(
            attention_np,
            cmap="jet",
            alpha=0.55,
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, col].set_title(f"Attention {name}", fontsize=10)
        axes[1, col].axis("off")

    for col in range(n_attention, n_cols):
        axes[1, col].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al YAML de configuración.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al checkpoint entrenado.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/attention_maps",
        help="Directorio de salida.",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=3,
        help="Número de imágenes a visualizar.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help=(
            "Split solicitado. Si se pide test pero DRIVE test no tiene "
            "1st_manual, se usará train automáticamente."
        ),
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"].get("name", "attention_unet")
    if model_name != "attention_unet":
        raise ValueError(
            "visualize_attention.py requiere un modelo attention_unet."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AttentionUNet(
        in_channels=cfg["model"].get("in_channels", 3),
        out_channels=cfg["model"].get("out_channels", 1),
        base_channels=cfg["model"].get("base_channels", 64),
        depth=cfg["model"].get("depth", 4),
    )

    state_dict = load_checkpoint_safely(args.checkpoint, device=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    captures, handles = register_attention_hooks(model)

    drive_root = cfg["data"].get("drive_root", "data/drive")
    split = resolve_drive_split(drive_root, args.split)

    img_size = cfg["data"].get("img_size", 512)
    use_clahe = cfg["data"].get("use_clahe", False)

    transform = ValTransform(
        img_size=img_size,
        use_clahe=use_clahe,
    )

    dataset = DriveDataset(
        root=drive_root,
        split=split,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"Visualizando Attention U-Net usando DRIVE split='{split}'.")
    print(f"Total de imágenes disponibles: {len(dataset)}")

    n_done = 0

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if n_done >= args.n_images:
                break

            image, mask, fov = unpack_batch(batch)

            image = image.to(device)
            mask = mask.detach().cpu()

            captures.clear()

            logits = model(image)
            probs = torch.sigmoid(logits).detach().cpu()

            pred_np = (probs.squeeze().numpy() > 0.5).astype(np.uint8)
            gt_np = mask.squeeze().numpy().astype(np.uint8)
            image_np = denormalize_image(image.squeeze(0))

            out_path = out_dir / f"attention_map_{n_done + 1:02d}.png"

            plot_attention_sample(
                image_np=image_np,
                gt_np=gt_np,
                pred_np=pred_np,
                captures=captures,
                out_path=out_path,
                title=f"Mapas de atención — DRIVE/{split} imagen #{idx + 1}",
            )

            print(f"✓ {out_path}")
            n_done += 1

    for handle in handles:
        handle.remove()

    print(f"\n✓ Mapas de atención guardados en: {out_dir}")


if __name__ == "__main__":
    main()