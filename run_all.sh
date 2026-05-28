#!/usr/bin/env bash
# run_all.sh — Pipeline completo de segmentación de vasos retinianos
#
# Único comando documentado para reproducir el proyecto de extremo a extremo:
#
#     bash run_all.sh
#
# Prerrequisitos (parte del environment spec):
#   1. Python 3.10+ con pip
#   2. GPU NVIDIA con drivers CUDA (recomendado; CPU funciona pero es ~30x más lento)
#   3. Token de Kaggle en ~/.kaggle/kaggle.json
#      (https://www.kaggle.com/settings → API → Create New Token)
#
# Tiempo estimado: ~3 h en L4 GPU, ~5-6 h en T4, ~10 min de descarga inicial.

set -e

# Garantizar que src/ esté en PYTHONPATH para que los imports
# `from data.dataset`, `from models...`, `from utils...` funcionen
# independientemente del entorno (Colab, servidor, local).
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

echo "=============================================="
echo "  Segmentación de Vasos Retinianos — Pipeline"
echo "=============================================="

# ── PASO 0: Entorno y dependencias ──────────────────────────────
echo ""
echo "[0/8] Verificando entorno e instalando dependencias..."

# Si torch ya está instalado y kaggle también, saltamos el pip install
if python -c "import torch, kaggle, scipy, sklearn, cv2, PIL, matplotlib, yaml, tqdm" 2>/dev/null; then
    echo "  ✓ Dependencias ya instaladas"
else
    # Intentamos pip install; si falla por PEP 668, usamos --break-system-packages
    if ! pip install -r requirements.txt -q 2>/dev/null; then
        echo "  Entorno externally-managed, instalando con --break-system-packages..."
        pip install -r requirements.txt -q --break-system-packages
    fi
    echo "  ✓ Dependencias instaladas"
fi

python - << 'PYEOF'
import torch
if torch.cuda.is_available():
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("  ⚠ No hay GPU. El entrenamiento será MUY lento (~30x).")
    print("  ⚠ Recomendado: Colab con GPU T4/L4 o servidor con NVIDIA GPU.")
PYEOF

# ── PASO 1: Descarga + organización de datasets ────────────────
echo ""
echo "[1/8] Descarga y organización de datasets desde Kaggle..."

if [ -n "${DATA_FROM_DRIVE}" ]; then
    echo "  ✓ Datasets montados desde Google Drive, saltando descarga"
else
    python scripts/setup_datasets.py --data_root data
fi

echo ""
echo "[1.5/8] Validando estructura y máscaras de datasets..."
python scripts/validate_datasets.py --img_size 512

# ── PASO 2: Entrenar modelo principal ──────────────────────────
echo ""
echo "[2/8] Entrenando Attention U-Net en DRIVE..."
echo "      Pérdida: BCE+Dice combinada | Augmentación: aggressive + CLAHE"
echo "      Tiempo estimado: ~45 min en GPU T4, ~25 min en L4"

python src/train.py --config configs/train_attention_unet.yaml
echo "  ✓ Attention U-Net entrenada"

# ── PASO 3: Estudio de ablación ────────────────────────────────
echo ""
echo "[3/8] Estudio de ablación..."

echo "  >> Ablación arquitectura: U-Net base (sin Attention Gates)"
python src/train.py --config configs/train_unet.yaml

echo "  >> Ablación pérdida: BCE pura"
python src/train.py --config configs/ablation_bce.yaml

echo "  >> Ablación pérdida: Dice pura"
python src/train.py --config configs/ablation_dice.yaml

echo "  >> Ablación pérdida: BCE + Dice combinada"
python src/train.py --config configs/ablation_combined.yaml

echo "  ✓ Estudio de ablación completado"

# ── PASO 4: Evaluación in-distribution + análisis ──────────────
echo ""
echo "[4/8] Evaluación en DRIVE..."

python scripts/evaluate.py \
    --config configs/train_attention_unet.yaml \
    --checkpoint outputs/attention_unet_drive/best_model.pt \
    --out_dir results/attention_unet_drive \
    --dataset drive

python scripts/evaluate.py \
    --config configs/train_unet.yaml \
    --checkpoint outputs/unet_base_drive/best_model.pt \
    --out_dir results/unet_base_drive \
    --dataset drive

# ── PASO 5: Análisis específicos ───────────────────────────────
echo ""
echo "[5/8] Análisis estratificado por grosor + mapas de atención..."

python scripts/vessel_thickness_analysis.py \
    --config configs/train_attention_unet.yaml \
    --checkpoint outputs/attention_unet_drive/best_model.pt \
    --out_dir results/vessel_thickness \
    --dataset drive

python scripts/visualize_attention.py \
    --config configs/train_attention_unet.yaml \
    --checkpoint outputs/attention_unet_drive/best_model.pt \
    --out_dir results/attention_maps \
    --n_images 5

python scripts/plot_training_curves.py \
    --history outputs/attention_unet_drive/training_history.json \
              outputs/unet_base_drive/training_history.json \
    --labels "Attention U-Net" "U-Net base" \
    --out_dir results/training_curves

# ── PASO 6: Generalización DRIVE → CHASE_DB1 ──────────────────
echo ""
echo "[6/8] Generalización de dominio DRIVE → CHASE_DB1..."

python scripts/domain_adaptation.py \
    --config configs/train_attention_unet.yaml \
    --checkpoint outputs/attention_unet_drive/best_model.pt \
    --chase_root data/chase_db1 \
    --out_dir results/domain_adaptation \
    --finetune_ratio 0.2

# ── PASO 7: Inter-anotador (STARE) ─────────────────────────────
echo ""
echo "[7/8] Concordancia entre anotadores (STARE)..."

# Solo correr si STARE tiene anotaciones de ambos expertos
if [ -d "data/stare/labels-ah" ] && [ -d "data/stare/labels-vk" ] && \
   [ "$(ls -A data/stare/labels-ah 2>/dev/null)" ] && \
   [ "$(ls -A data/stare/labels-vk 2>/dev/null)" ]; then
    python scripts/inter_annotator_stare.py \
        --stare_root data/stare \
        --out_dir results/inter_annotator \
        --config configs/train_attention_unet.yaml \
        --checkpoint outputs/attention_unet_drive/best_model.pt
else
    echo "  ⚠ STARE sin anotaciones AH/VK, paso omitido (es bonus, no obligatorio)"
fi

# ── PASO 8: Resumen final ──────────────────────────────────────
echo ""
echo "[8/8] Resumen de resultados"
echo ""
echo "=============================================="
echo "  RESULTADOS FINALES"
echo "=============================================="

python - << 'PYEOF'
import json, os

print()
print(f"  {'Experimento':<35} {'Sens':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
print(f"  {'-'*67}")

experimentos = [
    ("Attention U-Net (Combined)",
     "outputs/attention_unet_drive/test_metrics.json"),
    ("U-Net base (Combined)",
     "outputs/unet_base_drive/test_metrics.json"),
    ("Ablación BCE",
     "outputs/ablation_bce/test_metrics.json"),
    ("Ablación Dice",
     "outputs/ablation_dice/test_metrics.json"),
    ("Ablación Combined",
     "outputs/ablation_combined/test_metrics.json"),
]

for nombre, path in experimentos:
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        print(f"  {nombre:<35} "
              f"{m.get('sensibilidad', 0):>8.4f} "
              f"{m.get('especificidad', 0):>8.4f} "
              f"{m.get('f1', 0):>8.4f} "
              f"{m.get('auc_roc', 0):>8.4f}")

# Resultados de dominio
domain_path = "results/domain_adaptation/domain_adaptation_results.json"
if os.path.exists(domain_path):
    with open(domain_path) as f:
        d = json.load(f)
    print()
    print(f"  {'Generalización DRIVE→CHASE_DB1':<35} {'Sens':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
    print(f"  {'-'*67}")
    for nombre, key in [
        ("Sin adaptación",  "sin_adaptacion"),
        ("+ CLAHE",         "con_clahe"),
        ("+ Fine-tuning",   "con_finetune"),
    ]:
        if key in d:
            m = d[key]
            print(f"  {nombre:<35} "
                  f"{m.get('sensibilidad', 0):>8.4f} "
                  f"{m.get('especificidad', 0):>8.4f} "
                  f"{m.get('f1', 0):>8.4f} "
                  f"{m.get('auc_roc', 0):>8.4f}")

# Análisis por grosor
thickness_path = "results/vessel_thickness/vessel_thickness_results.json"
if os.path.exists(thickness_path):
    with open(thickness_path) as f:
        d = json.load(f)
    s = d.get("summary", {})
    print()
    print(f"  Sensibilidad por grosor de vaso:")
    print(f"  {'-'*67}")
    for cat in ("fino", "medio", "grueso"):
        if cat in s:
            print(f"    {cat:<15} sens = {s[cat]['sens_mean']:.4f} ± {s[cat]['sens_std']:.4f}")

# Inter-anotador
inter_path = "results/inter_annotator/inter_annotator_results.json"
if os.path.exists(inter_path):
    with open(inter_path) as f:
        d = json.load(f)
    s = d.get("summary", {})
    print()
    print(f"  Concordancia AH vs VK (STARE):")
    print(f"  {'-'*67}")
    for k in ("dice", "kappa"):
        if k in s:
            print(f"    {k:<15} = {s[k]['mean']:.4f} ± {s[k]['std']:.4f}")

print()
print("  Resultados detallados en: results/")
print("  Checkpoints y métricas:   outputs/")
PYEOF

echo ""
echo "=============================================="
echo "  Pipeline completado exitosamente"
echo "=============================================="