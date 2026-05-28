[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manadevelop/retinal_segmentation/blob/main/colab_runner.ipynb)

# Segmentación de Vasos Retinianos para Detección de Retinopatía Diabética

Proyecto correspondiente a la **Pregunta 2 del Examen Parcial** del curso **Redes Neuronales y Aprendizaje Profundo** de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería.

**Docente:** Ph.D. Aldo Camargo  
**Repositorio:** <https://github.com/manadevelop/retinal_segmentation>

**Integrantes:**

- Victor Fernando Montes Jaramillo
- Alex Celestino León Pacheco
- Edwin Jhon Minchán Ramos
- Marco Antonio Nina Aguilar

---

## 1. Descripción del problema

La retinopatía diabética es una de las principales causas de ceguera prevenible. Su detección temprana se apoya en el análisis de fotografías de fondo de ojo, donde la segmentación de la red vascular permite estudiar características como calibre, tortuosidad, ramificaciones y presencia de vasos finos.

Este proyecto implementa un pipeline completo de **segmentación binaria píxel a píxel** de vasos retinianos usando **U-Net** y **Attention U-Net** implementadas en PyTorch. El objetivo es segmentar cada píxel de una imagen de fondo de ojo como:

- **vaso sanguíneo**, o
- **fondo / no vaso**.

Además del entrenamiento en DRIVE, el proyecto analiza la degradación del rendimiento al evaluar sobre CHASE_DB1, un dominio diferente, e implementa estrategias de mitigación como **CLAHE** y **fine-tuning** sobre una pequeña fracción del dominio objetivo.

---

## 2. Entregables cubiertos

| # | Entregable de la Pregunta 2 | Estado en el proyecto |
|---|---|---|
| 1 | Implementación personalizada de U-Net en PyTorch | `src/models/unet.py` y `src/models/attention_unet.py` |
| 2 | Ablación de arquitectura o entrenamiento | U-Net vs Attention U-Net; BCE vs Dice vs BCE+Dice |
| 3 | Evaluación en DRIVE con Sensibilidad, Especificidad, F1 y AUC-ROC | `outputs/*/test_metrics.json` y `scripts/evaluate.py` |
| 4 | Generalización DRIVE → CHASE_DB1 | `scripts/domain_adaptation.py` |
| 5 | Análisis de vasos finos vs gruesos | `scripts/vessel_thickness_analysis.py` |
| 6 | Estrategia de mitigación de dominio | CLAHE y fine-tuning con 25% de CHASE_DB1 |
| 7 | Ejecución extremo a extremo con un único comando | `bash run_all.sh` |
| 8 | Especificación de entorno | `requirements.txt` |

---

## 3. Datasets utilizados

Los datasets **no se versionan en GitHub**. Deben colocarse manualmente en Google Drive o en la carpeta local `data/`, según el modo de ejecución.

### 3.1 DRIVE

Fuente oficial: <https://drive.grand-challenge.org/>

En este proyecto se usó una copia manual organizada en Google Drive con la siguiente estructura real:

```text
EP01/pregunta2/datasets/DRIVE/
├── datasets/
│   ├── training/
│   │   ├── images/       # imágenes .tif
│   │   ├── mask/         # máscaras FOV .gif
│   │   └── 1st_manual/   # máscaras manuales de vasos .gif
│   └── test/
│       ├── images/       # imágenes .tif
│       └── mask/         # máscaras FOV .gif
```

> Nota metodológica: la partición `test` de DRIVE en esta copia no contiene `1st_manual`. La carpeta `mask/` corresponde al campo de visión retinal, no al ground truth de vasos. Por eso el pipeline usa un holdout interno desde `training/` para calcular métricas locales válidas cuando no existe `test/1st_manual/`.

### 3.2 CHASE_DB1

Fuente institucional: <https://researchdata.kingston.ac.uk/96/>

CHASE_DB1 fue descargado y organizado manualmente. La estructura usada fue:

```text
EP01/pregunta2/datasets/CHASE_DB1/new/chase/test/test/
├── images/       # imágenes .tif
├── mask/         # máscaras FOV .tif
├── 1st_manual/   # máscaras manuales .tif
└── 2nd_manual/   # segundo anotador .png
```

En el pipeline se mapea a:

```text
data/chase_db1/
├── images/       -> .../CHASE_DB1/new/chase/test/test/images
├── labels/       -> .../CHASE_DB1/new/chase/test/test/1st_manual
├── mask/         -> .../CHASE_DB1/new/chase/test/test/mask
└── labels-2nd/   -> .../CHASE_DB1/new/chase/test/test/2nd_manual
```

### 3.3 STARE

Fuente oficial: <https://cecas.clemson.edu/~ahoover/stare/>

La estructura usada fue:

```text
EP01/pregunta2/datasets/STARE/
├── images/
└── masks/
```

En el pipeline se mapea a:

```text
data/stare/
├── images/       -> .../STARE/images
├── masks/        -> .../STARE/masks
└── labels-ah/    -> .../STARE/masks
```

Si se dispone también de anotaciones VK, pueden agregarse como `data/stare/labels-vk/` para ejecutar el análisis inter-anotador. En la ejecución final de este proyecto, el análisis STARE AH/VK se considera complementario.

---

## 4. Estructura esperada del repositorio

```text
retinal_segmentation/
├── README.md
├── requirements.txt
├── run_all.sh
├── colab_runner.ipynb
├── .gitignore
├── configs/
│   ├── train_attention_unet.yaml
│   ├── train_unet.yaml
│   ├── ablation_bce.yaml
│   ├── ablation_dice.yaml
│   └── ablation_combined.yaml
├── src/
│   ├── train.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   └── attention_unet.py
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py
│       ├── logger.py
│       ├── losses.py
│       ├── metrics.py
│       └── trainer.py
├── scripts/
│   ├── validate_datasets.py
│   ├── evaluate.py
│   ├── domain_adaptation.py
│   ├── vessel_thickness_analysis.py
│   ├── visualize_attention.py
│   ├── plot_training_curves.py
│   └── inter_annotator_stare.py
├── data/       # no se sube a GitHub
├── outputs/    # no se sube a GitHub
├── results/    # no se sube a GitHub
└── reports/
```

Las carpetas `data/`, `outputs/` y `results/` están excluidas mediante `.gitignore` porque contienen datasets, checkpoints, métricas, figuras y artefactos generados.

---

## 5. Ejecución recomendada en Google Colab

La forma recomendada de ejecutar el proyecto es mediante:

```text
colab_runner.ipynb
```

### 5.1 Preparar Google Drive

Antes de ejecutar, coloca los datasets en Google Drive así:

```text
MyDrive/
└── EP01/
    └── pregunta2/
        └── datasets/
            ├── DRIVE/
            ├── CHASE_DB1/
            └── STARE/
```

La estructura interna debe coincidir con lo indicado en la sección de datasets.

### 5.2 Abrir el notebook

Puedes abrir el notebook usando el botón **Open in Colab** de la parte superior del README o entrando manualmente a:

```text
https://colab.research.google.com/github/manadevelop/retinal_segmentation/blob/main/colab_runner.ipynb
```

### 5.3 Configurar GPU

En Colab:

```text
Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU
```

Se recomienda usar **L4** o **T4**.

### 5.4 Ejecutar celdas del notebook

El notebook realiza automáticamente:

1. Montaje de Google Drive.
2. Clonación o actualización del repositorio.
3. Creación de symlinks desde `data/` hacia las carpetas en Google Drive.
4. Validación de estructura y máscaras.
5. Ejecución del pipeline completo:

```bash
bash run_all.sh
```

6. Verificación de archivos generados.
7. Respaldo de resultados en Google Drive.
8. Visualización resumida de métricas y mapas de atención.

### 5.5 Dónde se guardan los resultados en Colab

Durante la ejecución, los artefactos se generan en:

```text
/content/retinal_segmentation/outputs/
/content/retinal_segmentation/results/
```

Luego el notebook los respalda en Google Drive:

```text
/content/drive/MyDrive/EP01/pregunta2/resultados/outputs/
/content/drive/MyDrive/EP01/pregunta2/resultados/results/
```

---

## 6. Ejecución local o en servidor

Si se ejecuta fuera de Colab, primero instala dependencias:

```bash
pip install -r requirements.txt
```

Luego asegúrate de tener los datasets organizados así:

```text
data/
├── drive/
├── chase_db1/
└── stare/
```

Finalmente ejecuta:

```bash
bash run_all.sh
```

---

## 7. Qué hace `run_all.sh`

El archivo `run_all.sh` es el comando único de reproducción. Ejecuta los siguientes pasos:

```text
[0/8] Verifica entorno e instala dependencias
[1/8] Valida datasets montados en data/
[2/8] Entrena Attention U-Net en DRIVE
[3/8] Ejecuta ablaciones: U-Net base, BCE, Dice, BCE+Dice
[4/8] Evalúa modelos en DRIVE local
[5/8] Genera análisis por grosor, mapas de atención y curvas
[6/8] Evalúa generalización DRIVE → CHASE_DB1
[7/8] Ejecuta análisis inter-anotador STARE si existen AH/VK
[8/8] Imprime resumen final de resultados
```

El paso de dominio ejecuta:

```bash
python scripts/domain_adaptation.py \
  --config configs/train_attention_unet.yaml \
  --checkpoint outputs/attention_unet_drive/best_model.pt \
  --chase_root data/chase_db1 \
  --out_dir results/domain_adaptation \
  --finetune_ratio 0.25
```

---

## 8. Resultados generados

### 8.1 Carpeta `outputs/`

Contiene checkpoints, métricas JSON e historiales de entrenamiento:

```text
outputs/
├── attention_unet_drive/
│   ├── best_model.pt
│   ├── test_metrics.json
│   └── training_history.json
├── unet_base_drive/
│   ├── best_model.pt
│   ├── test_metrics.json
│   └── training_history.json
├── ablation_bce/
├── ablation_dice/
└── ablation_combined/
```

### 8.2 Carpeta `results/`

Contiene figuras, análisis visuales y reportes derivados:

```text
results/
├── attention_unet_drive/
│   ├── predictions.png
│   ├── roc_curve.png
│   └── failure_analysis.png
├── unet_base_drive/
├── attention_maps/
│   ├── attention_map_01.png
│   ├── attention_map_02.png
│   └── attention_map_03.png
├── vessel_thickness/
│   ├── vessel_thickness_results.json
│   ├── sensitivity_by_thickness.png
│   └── thickness_visualization.png
├── training_curves/
└── domain_adaptation/
```

---

## 9. Resultados finales obtenidos

### 9.1 Comparación de arquitecturas en DRIVE local

| Modelo | Sensibilidad | Especificidad | Precisión | F1 / Dice | IoU | Accuracy | AUC-ROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention U-Net | 0.7598 | 0.9360 | 0.6383 | 0.6937 | 0.5327 | 0.9129 | 0.9258 |
| U-Net base | 0.7701 | 0.9337 | 0.6353 | 0.6962 | 0.5356 | 0.9123 | 0.9254 |

### 9.2 Ablación de pérdidas

| Pérdida | Sensibilidad | Especificidad | Precisión | F1 / Dice | IoU | Accuracy | AUC-ROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| BCE | 0.8732 | 0.8465 | 0.4603 | 0.6026 | 0.4318 | 0.8499 | 0.9297 |
| Dice | 0.8508 | 0.8156 | 0.4082 | 0.5516 | 0.3812 | 0.8202 | 0.9076 |
| BCE+Dice | 0.8754 | 0.8439 | 0.4571 | 0.6003 | 0.4295 | 0.8480 | 0.9296 |

### 9.3 Generalización DRIVE → CHASE_DB1

| Configuración | Sensibilidad | Especificidad | F1 / Dice | AUC-ROC |
|---|---:|---:|---:|---:|
| Sin adaptación | 0.1094 | 0.9961 | 0.1833 | 0.7246 |
| + CLAHE | 0.3976 | 0.9766 | 0.4789 | 0.7827 |
| + Fine-tuning 25% CHASE_DB1 | 0.7690 | 0.9431 | 0.6500 | 0.9229 |

### 9.4 Análisis por grosor de vaso

| Tipo de vaso | Definición | Sensibilidad media |
|---|---|---:|
| Fino | radio ≤ 2 px | 0.7526 ± 0.0539 |
| Medio | 2 < radio ≤ 4 px | 0.9795 ± 0.0320 |
| Grueso | radio > 4 px | 0.9498 ± 0.1334 |

---

## 10. Interpretación resumida

- U-Net base obtuvo el mejor F1 local, aunque la diferencia con Attention U-Net fue mínima.
- Attention U-Net fue útil para interpretación mediante mapas de atención.
- BCE y BCE+Dice aumentaron la sensibilidad, pero redujeron precisión y especificidad, evidenciando sobresegmentación.
- Dice puro fue la pérdida menos competitiva en F1 e IoU.
- La generalización directa DRIVE → CHASE_DB1 fue débil, con F1 = 0.1833.
- CLAHE redujo parte de la brecha de dominio, elevando F1 a 0.4789.
- Fine-tuning con 25% de CHASE_DB1 logró F1 = 0.6500 y AUC = 0.9229.
- Los vasos finos fueron los más difíciles de segmentar por el downsampling de U-Net y la baja representación de capilares en píxeles.

---

## 11. Reproducibilidad

- Semilla fija: `42`.
- Configuración por YAML en `configs/`.
- Métricas guardadas como JSON.
- Resultados y figuras generados automáticamente.
- El pipeline puede ejecutarse con:

```bash
bash run_all.sh
```

En Colab, se recomienda usar `colab_runner.ipynb`, que prepara rutas, symlinks y respaldo de resultados.

---

## 12. Archivos que no deben subirse a GitHub

El repositorio debe excluir:

```text
data/
outputs/
results/
logs/
*.pt
*.pth
*.ckpt
```

Estas carpetas se generan o se montan durante la ejecución y pueden ser muy pesadas.

---

## 13. Referencias principales

1. Ronneberger, O., Fischer, P., Brox, T. **U-Net: Convolutional Networks for Biomedical Image Segmentation**. MICCAI, 2015.
2. Oktay, O. et al. **Attention U-Net: Learning Where to Look for the Pancreas**. MIDL, 2018.
3. Staal, J. et al. **Ridge-Based Vessel Segmentation in Color Images of the Retina**. IEEE Transactions on Medical Imaging, 2004.
4. Hoover, A., Kouznetsova, V., Goldbaum, M. **Locating Blood Vessels in Retinal Images by Piecewise Threshold Probing of a Matched Filter Response**. IEEE Transactions on Medical Imaging, 2000.
5. Fraz, M.M. et al. **An Ensemble Classification-Based Approach Applied to Retinal Blood Vessel Segmentation**. IEEE Transactions on Biomedical Engineering, 2012.
6. DRIVE Grand Challenge. <https://drive.grand-challenge.org/>
7. STARE Project. <https://cecas.clemson.edu/~ahoover/stare/>
8. CHASE_DB1 / Kingston Research Data. <https://researchdata.kingston.ac.uk/96/>

---

## 14. Licencia

Este repositorio contiene código académico desarrollado para el curso de Redes Neuronales y Aprendizaje Profundo. Los datasets conservan sus propias condiciones de uso y licencias.
