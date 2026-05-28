#!/usr/bin/env bash
# run_all.sh — Pipeline completo de segmentación de vasos retinianos
#
# Único comando documentado para reproducir el proyecto de extremo a extremo:
#
#     bash run_all.sh
#
# Modos de uso:
#
# 1) Usando datasets montados desde Google Drive:
#        export DATA_FROM_DRIVE=1
#        bash run_all.sh
#
# 2) Usando descarga automática desde Kaggle:
#        bash run_all.sh
#
# En Colab, si los datasets ya están montados mediante symlinks en data/,
# usar siempre DATA_FROM_DRIVE=1 para evitar autenticación Kaggle.
#
# Requisitos:
#   - Python 3.10+
#   - requirements.txt
#   - GPU NVIDIA recomendada
#   - Estructura esperada:
#       data/drive
#       data/chase_db1
#       data/stare

set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Configuración general
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

ATTENTION_CONFIG="configs/train_attention_unet.yaml"
UNET_CONFIG="configs/train_unet.yaml"

ATTENTION_CKPT="outputs/attention_unet_drive/best_model.pt"
UNET_CKPT="outputs/unet_base_drive/best_model.pt"

CHASE_ROOT="data/chase_db1"
DOMAIN_OUT="results/domain_adaptation"

IMG_SIZE=512
FINETUNE_RATIO=0.25

echo "=============================================="
echo "  Segmentación de Vasos Retinianos — Pipeline"
echo "=============================================="
echo ""
echo "Proyecto     : ${PROJECT_ROOT}"
echo "PYTHONPATH   : ${PYTHONPATH}"
echo "DATA_FROM_DRIVE: ${DATA_FROM_DRIVE:-0}"
echo ""

# ──────────────────────────────────────────────────────────────
# PASO 0: Entorno y dependencias
# ──────────────────────────────────────────────────────────────

echo "[0/8] Verificando entorno e instalando dependencias..."

# Evitamos importar kaggle cuando DATA_FROM_DRIVE=1, porque el import puede
# emitir advertencias de autenticación aunque no se vaya a descargar nada.
if [ -n "${DATA_FROM_DRIVE:-}" ]; then
    if python - << 'PYEOF' 2>/dev/null
import torch
import scipy
import sklearn
import cv2
import PIL
import matplotlib
import yaml
import tqdm
PYEOF
    then
        echo "  ✓ Dependencias principales ya instaladas"
    else
        echo "  Instalando dependencias desde requirements.txt..."
        if ! pip install -r requirements.txt -q 2>/dev/null; then
            echo "  Entorno externally-managed, instalando con --break-system-packages..."
            pip install -r requirements.txt -q --break-system-packages
        fi
        echo "  ✓ Dependencias instaladas"
    fi
else
    if python - << 'PYEOF' 2>/dev/null
import torch
import kaggle
import scipy
import sklearn
import cv2
import PIL
import matplotlib
import yaml
import tqdm
PYEOF
    then
        echo "  ✓ Dependencias ya instaladas"
    else
        echo "  Instalando dependencias desde requirements.txt..."
        if ! pip install -r requirements.txt -q 2>/dev/null; then
            echo "  Entorno externally-managed, instalando con --break-system-packages..."
            pip install -r requirements.txt -q --break-system-packages
        fi
        echo "  ✓ Dependencias instaladas"
    fi
fi

python - << 'PYEOF'
import torch

if torch.cuda.is_available():
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("  ⚠ No hay GPU. El entrenamiento será muy lento.")
    print("  ⚠ Recomendado: Colab con GPU T4/L4 o servidor con NVIDIA GPU.")
PYEOF

# ──────────────────────────────────────────────────────────────
# PASO 1: Datasets
# ──────────────────────────────────────────────────────────────

echo ""
echo "[1/8] Descarga y organización de datasets..."

if [ -n "${DATA_FROM_DRIVE:-}" ]; then
    echo "  ✓ DATA_FROM_DRIVE=1 detectado"
    echo "  ✓ Datasets montados desde Google Drive; se omite descarga Kaggle"
else
    echo "  DATA_FROM_DRIVE no está configurado; intentando descarga desde Kaggle..."
    python scripts/setup_datasets.py --data_root data
fi

echo ""
echo "[1.5/8] Validando estructura y máscaras de datasets..."
python scripts/validate_datasets.py --img_size "${IMG_SIZE}"

# ──────────────────────────────────────────────────────────────
# PASO 2: Modelo principal
# ──────────────────────────────────────────────────────────────

echo ""
echo "[2/8] Entrenando Attention U-Net en DRIVE..."
echo "      Pérdida: BCE+Dice combinada | Augmentación: aggressive + CLAHE"
echo "      Tiempo estimado: ~45 min en GPU T4, ~25 min en L4"

python src/train.py --config "${ATTENTION_CONFIG}"

if [ ! -f "${ATTENTION_CKPT}" ]; then
    echo "ERROR: No se generó checkpoint esperado: ${ATTENTION_CKPT}"
    exit 1
fi

echo "  ✓ Attention U-Net entrenada"

# ──────────────────────────────────────────────────────────────
# PASO 3: Ablaciones
# ──────────────────────────────────────────────────────────────

echo ""
echo "[3/8] Estudio de ablación..."

echo "  >> Ablación arquitectura: U-Net base"
python src/train.py --config "${UNET_CONFIG}"

echo "  >> Ablación pérdida: BCE pura"
python src/train.py --config configs/ablation_bce.yaml

echo "  >> Ablación pérdida: Dice pura"
python src/train.py --config configs/ablation_dice.yaml

echo "  >> Ablación pérdida: BCE + Dice combinada"
python src/train.py --config configs/ablation_combined.yaml

echo "  ✓ Estudio de ablación completado"

# ──────────────────────────────────────────────────────────────
# PASO 4: Evaluación in-distribution
# ──────────────────────────────────────────────────────────────

echo ""
echo "[4/8] Evaluación local en DRIVE..."

python scripts/evaluate.py \
    --config "${ATTENTION_CONFIG}" \
    --checkpoint "${ATTENTION_CKPT}" \
    --out_dir results/attention_unet_drive \
    --dataset drive

python scripts/evaluate.py \
    --config "${UNET_CONFIG}" \
    --checkpoint "${UNET_CKPT}" \
    --out_dir results/unet_base_drive \
    --dataset drive

echo "  ✓ Evaluación local completada"

# ──────────────────────────────────────────────────────────────
# PASO 5: Análisis específicos
# ──────────────────────────────────────────────────────────────

echo ""
echo "[5/8] Análisis estratificado por grosor + mapas de atención..."

python scripts/vessel_thickness_analysis.py \
    --config "${ATTENTION_CONFIG}" \
    --checkpoint "${ATTENTION_CKPT}" \
    --out_dir results/vessel_thickness \
    --dataset drive

python scripts/visualize_attention.py \
    --config "${ATTENTION_CONFIG}" \
    --checkpoint "${ATTENTION_CKPT}" \
    --out_dir results/attention_maps \
    --n_images 5

python scripts/plot_training_curves.py \
    --history outputs/attention_unet_drive/training_history.json \
              outputs/unet_base_drive/training_history.json \
    --labels "Attention U-Net" "U-Net base" \
    --out_dir results/training_curves

echo "  ✓ Análisis específicos completados"

# ──────────────────────────────────────────────────────────────
# PASO 6: Generalización DRIVE → CHASE_DB1
# ──────────────────────────────────────────────────────────────

echo ""
echo "[6/8] Generalización de dominio DRIVE → CHASE_DB1..."

if [ ! -d "${CHASE_ROOT}" ]; then
    echo "ERROR: No existe ${CHASE_ROOT}"
    echo "Verifica que los symlinks de CHASE_DB1 estén creados:"
    echo "  data/chase_db1/images"
    echo "  data/chase_db1/labels"
    echo "  data/chase_db1/mask"
    exit 1
fi

python scripts/domain_adaptation.py \
    --config "${ATTENTION_CONFIG}" \
    --checkpoint "${ATTENTION_CKPT}" \
    --chase_root "${CHASE_ROOT}" \
    --out_dir "${DOMAIN_OUT}" \
    --finetune_ratio "${FINETUNE_RATIO}"

echo "  ✓ Generalización DRIVE → CHASE_DB1 completada"

# ──────────────────────────────────────────────────────────────
# PASO 7: Inter-anotador STARE
# ──────────────────────────────────────────────────────────────

echo ""
echo "[7/8] Concordancia entre anotadores STARE..."

if [ -d "data/stare/labels-ah" ] && [ -d "data/stare/labels-vk" ] && \
   [ "$(ls -A data/stare/labels-ah 2>/dev/null || true)" ] && \
   [ "$(ls -A data/stare/labels-vk 2>/dev/null || true)" ]; then

    python scripts/inter_annotator_stare.py \
        --stare_root data/stare \
        --out_dir results/inter_annotator \
        --config "${ATTENTION_CONFIG}" \
        --checkpoint "${ATTENTION_CKPT}"

    echo "  ✓ Concordancia inter-anotador completada"
else
    echo "  ⚠ STARE sin anotaciones AH/VK completas; paso omitido"
    echo "  ⚠ Este paso es complementario y no bloquea la Pregunta 2"
fi

# ──────────────────────────────────────────────────────────────
# PASO 8: Resumen final
# ──────────────────────────────────────────────────────────────

echo ""
echo "[8/8] Resumen de resultados"
echo ""
echo "=============================================="
echo "  RESULTADOS FINALES"
echo "=============================================="

python - << 'PYEOF'
import json
import math
from pathlib import Path


def fmt(x):
    if x is None:
        return "   n/a  "
    try:
        x = float(x)
        if math.isnan(x):
            return "   n/a  "
        return f"{x:8.4f}"
    except Exception:
        return "   n/a  "


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


print()
print(f"  {'Experimento':<35} {'Sens':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
print(f"  {'-'*67}")

experimentos = [
    ("Attention U-Net", "outputs/attention_unet_drive/test_metrics.json"),
    ("U-Net base", "outputs/unet_base_drive/test_metrics.json"),
    ("Ablación BCE", "outputs/ablation_bce/test_metrics.json"),
    ("Ablación Dice", "outputs/ablation_dice/test_metrics.json"),
    ("Ablación BCE+Dice", "outputs/ablation_combined/test_metrics.json"),
]

for nombre, path in experimentos:
    m = load_json(path)
    if m is None:
        print(f"  {nombre:<35} {'FALTA':>8}")
        continue

    print(
        f"  {nombre:<35} "
        f"{fmt(m.get('sensibilidad'))} "
        f"{fmt(m.get('especificidad'))} "
        f"{fmt(m.get('f1'))} "
        f"{fmt(m.get('auc_roc'))}"
    )

# Resultados de dominio.
# Se intenta leer nombres comunes; si no existe JSON, se informa revisar log.
candidate_domain_jsons = [
    "results/domain_adaptation/domain_adaptation_results.json",
    "results/domain_adaptation/results.json",
    "results/domain_adaptation/metrics.json",
    "results/domain_adaptation/summary.json",
]

domain_data = None
domain_path = None

for p in candidate_domain_jsons:
    d = load_json(p)
    if d is not None:
        domain_data = d
        domain_path = p
        break

print()
print(f"  {'Generalización DRIVE→CHASE_DB1':<35} {'Sens':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
print(f"  {'-'*67}")

if domain_data is not None:
    # Soporta varias convenciones de nombres.
    possible_keys = [
        ("Sin adaptación", ["sin_adaptacion", "directo", "no_adaptation", "baseline"]),
        ("+ CLAHE", ["con_clahe", "clahe", "with_clahe"]),
        ("+ Fine-tuning", ["con_finetune", "fine_tuning", "finetune", "with_finetune"]),
    ]

    printed = False

    for label, keys in possible_keys:
        for key in keys:
            if key in domain_data and isinstance(domain_data[key], dict):
                m = domain_data[key]
                print(
                    f"  {label:<35} "
                    f"{fmt(m.get('sensibilidad'))} "
                    f"{fmt(m.get('especificidad'))} "
                    f"{fmt(m.get('f1'))} "
                    f"{fmt(m.get('auc_roc'))}"
                )
                printed = True
                break

    if not printed:
        print(f"  Se encontró JSON de dominio, pero con estructura no estándar: {domain_path}")
else:
    print("  No se encontró JSON estándar de dominio.")
    print("  Revisar log y carpeta: results/domain_adaptation")

# Análisis por grosor.
thickness_path = Path("results/vessel_thickness/vessel_thickness_results.json")
thickness_data = load_json(thickness_path)

if thickness_data is not None:
    print()
    print("  Sensibilidad por grosor de vaso:")
    print(f"  {'-'*67}")

    summary = thickness_data.get("summary", thickness_data)

    for cat in ("fino", "medio", "grueso"):
        if cat not in summary:
            continue

        item = summary[cat]
        mean = item.get("sens_mean", item.get("mean", None))
        std = item.get("sens_std", item.get("std", None))

        print(f"    {cat:<15} sens = {fmt(mean).strip()} ± {fmt(std).strip()}")

# Inter-anotador.
inter_path = Path("results/inter_annotator/inter_annotator_results.json")
inter_data = load_json(inter_path)

if inter_data is not None:
    print()
    print("  Concordancia AH vs VK STARE:")
    print(f"  {'-'*67}")

    summary = inter_data.get("summary", inter_data)

    for k in ("dice", "kappa"):
        if k in summary:
            item = summary[k]
            print(
                f"    {k:<15} = "
                f"{fmt(item.get('mean')).strip()} ± {fmt(item.get('std')).strip()}"
            )

print()
print("  Resultados detallados en: results/")
print("  Checkpoints y métricas:   outputs/")
PYEOF

echo ""
echo "=============================================="
echo "  Pipeline completado exitosamente"
echo "=============================================="