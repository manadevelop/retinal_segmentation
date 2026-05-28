[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manadevelop/retinal_segmentation/blob/main/colab_runner.ipynb)

# Segmentación de Vasos Retinianos para Detección de Retinopatía Diabética

## Pregunta 2 — Examen Parcial

**Curso:** Redes Neuronales y Aprendizaje Profundo    
**Docente:** Ph.D. Aldo Camargo  
**Universidad Nacional de Ingeniería — Maestría en Inteligencia Artificial**

**Integrantes:**

- Victor Fernando Montes Jaramillo
- Alex Celestino León Pacheco
- Edwin Jhon Minchán Ramos
- Marco Antonio Nina Aguilar

**Repositorio:** <https://github.com/manadevelop/retinal_segmentation>

---

## Enunciado del problema

La retinopatía diabética es una de las principales causas mundiales de ceguera prevenible. Su tamizaje temprano depende de la segmentación precisa de la red vascular en fotografías de fondo de ojo (imágenes de fundus): un problema de **predicción densa**, donde el modelo debe asignar a cada píxel una etiqueta binaria (*vaso* o *fondo*).

Este proyecto implementa una **U-Net** y una **Attention U-Net** desde cero en PyTorch para resolver esta tarea, con dos focos:

1. **Desempeño en distribución**: alcanzar un F1/Dice competitivo en DRIVE, el benchmark de referencia, con un estudio de ablación que justifique las decisiones arquitectónicas y de función de pérdida.
2. **Generalización entre dominios**: cuantificar la degradación del modelo al evaluarlo en imágenes adquiridas con una cámara de fundus distinta (CHASE_DB1) e implementar al menos una estrategia de mitigación de dicha brecha (preprocesamiento CLAHE, *test-time augmentation* o ajuste fino sobre el dominio objetivo).

---

## Conjuntos de datos

Se utilizan los tres datasets públicos de referencia indicados en el examen:

### DRIVE — Digital Retinal Images for Vessel Extraction
- **40 imágenes** de fundus con máscaras manuales de vasos y una división fija de entrenamiento/prueba (20/20).
- Benchmark estándar del área.
- Fuente oficial: <https://drive.grand-challenge.org/>
- Kaggle: <https://www.kaggle.com/datasets/umairinayat/retinal-vessel-segmentation-datasets>

### STARE — Structured Analysis of the Retina
- **20 imágenes** con **dos conjuntos independientes de anotaciones** (Adam Hoover y Valentina Kouznetsova).
- Permite estimar la concordancia entre anotadores como un *techo humano* contra el cual comparar el modelo.
- Fuente oficial: <https://cecas.clemson.edu/~ahoover/stare/>
- Kaggle: <https://www.kaggle.com/datasets/umairinayat/retinal-vessel-segmentation-datasets>

### CHASE_DB1
- **28 imágenes** adquiridas con una cámara de fundus diferente a DRIVE.
- Ideal para el experimento de generalización entre conjuntos (entregable #4).
- Fuente oficial original: <https://blogs.kingston.ac.uk/retinal/chasedb1/> (sitio actualmente caído)
- Espejo institucional: <https://researchdata.kingston.ac.uk/96/>
- Kaggle: <https://www.kaggle.com/datasets/khoongweihao/chasedb1>


### Nota crítica sobre DRIVE test y métricas locales

La página oficial de DRIVE indica que el conjunto completo tiene 40 imágenes divididas en 20 de entrenamiento y 20 de prueba, pero también aclara que para los casos de prueba no se publican las anotaciones manuales; las predicciones deben enviarse al sitio para evaluación oficial. Por eso, este repositorio implementa una política reproducible:

- Si `data/drive/test/1st_manual/` existe, el pipeline usa el split fijo oficial para evaluación.
- Si `data/drive/test/1st_manual/` no existe, el pipeline no crea máscaras vacías ni reporta métricas falsas. En su lugar, usa un holdout interno estratificado desde `data/drive/training/` para obtener métricas locales válidas con ground truth. El conjunto `test/images` puede usarse para generar predicciones, pero no para calcular sensibilidad/F1/AUC sin anotaciones.

Antes de entrenar, `run_all.sh` ejecuta `scripts/validate_datasets.py`, que verifica que las máscaras de vasos tengan píxeles positivos (`mask_sum > 0`). Esto evita resultados inválidos como `F1=0` y `AUC=nan` causados por máscaras faltantes o mal binarizadas.

### Justificación del uso de espejos en Kaggle

El examen exige que el código se ejecute *de extremo a extremo con un único comando*. Las fuentes oficiales de los tres datasets presentan obstáculos que rompen este requisito al automatizar la descarga:

- **DRIVE** requiere registro manual en grand-challenge.org y aceptación de términos antes de habilitar la descarga; no expone una URL pública directa.
- El sitio histórico de **CHASE_DB1** (`blogs.kingston.ac.uk/retinal/chasedb1/`) está caído; existe un espejo institucional, pero una sola URL de respaldo es frágil.
- **STARE** distribuye los archivos `.ppm` uno por uno desde el servidor de Clemson; no hay un zip único.

Por estas razones, el script `scripts/setup_datasets.py` descarga los datasets desde **espejos públicos en Kaggle** que la comunidad académica utiliza habitualmente. Los tres mirrors preservan el contenido y la licencia de las fuentes originales; solo cambia la mecánica de descarga. Esto permite cumplir la consigna del comando único sin perder trazabilidad académica del origen de los datos.

| Dataset    | Mirror usado (Kaggle)                                              | Fuente oficial                                |
|------------|--------------------------------------------------------------------|-----------------------------------------------|
| DRIVE      | `umairinayat/retinal-vessel-segmentation-datasets`                 | grand-challenge.org                           |
| STARE      | `umairinayat/retinal-vessel-segmentation-datasets`                 | cecas.clemson.edu/~ahoover/stare/             |
| CHASE_DB1  | `khoongweihao/chasedb1`                                            | researchdata.kingston.ac.uk/96/               |

---

## Entregables requeridos

Los seis entregables que pide el examen y su ubicación en este repositorio:

| # | Entregable                                                                    | Implementación                                                              |
|---|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| 1 | Implementación personalizada de U-Net en PyTorch, sin copiar bibliotecas      | `src/models/unet.py` (U-Net base) + `src/models/attention_unet.py` (variante con Attention Gates) |
| 2 | Estudio de ablación sobre ≥ 2 decisiones arquitectónicas / de entrenamiento   | Arquitectura: U-Net vs Attention U-Net. Pérdida: BCE / Dice / BCE+Dice combinada. Configs en `configs/*.yaml` |
| 3 | Evaluación in-distribution sobre DRIVE (sensibilidad, especificidad, F1, AUC) | `scripts/evaluate.py` → `results/<exp>/test_metrics.json` + figuras de ROC y análisis de fallos |
| 4 | Experimento de generalización entre conjuntos: entrenar en DRIVE, evaluar en CHASE_DB1; reportar brecha y analizar causa | `scripts/domain_adaptation.py` → `results/domain_adaptation/domain_adaptation_results.json` |
| 5 | Análisis cualitativo de fallos por tipo de vaso (capilares finos vs arterias grandes) con explicación arquitectónica | `scripts/vessel_thickness_analysis.py` (estratificación por radio vía `distance_transform_edt`) + `scripts/visualize_attention.py` (mapas de atención de cada nivel del decoder) |
| 6 | Estrategia de adaptación de dominio o preprocesamiento, con evidencia de su efecto sobre la brecha | Dos estrategias implementadas en `scripts/domain_adaptation.py`: preprocesamiento CLAHE y fine-tuning sobre subconjunto del dominio objetivo |

**Análisis bonus** (más allá de los entregables mínimos):

- Concordancia entre anotadores en STARE (Dice, IoU, Cohen's kappa) como techo humano: `scripts/inter_annotator_stare.py`.
- Visualización de los mapas de atención α de cada `AttentionGate` superpuestos sobre la imagen de entrada: `scripts/visualize_attention.py`.

---

## Resumen del proyecto

| Aspecto                | Detalle                                              |
|------------------------|------------------------------------------------------|
| Tarea                  | Segmentación binaria píxel-a-píxel                   |
| Arquitecturas          | U-Net (base) + Attention U-Net (principal)           |
| Dataset principal      | DRIVE (40 imágenes, split fijo 20/20)                |
| Datasets adicionales   | CHASE_DB1 (cross-domain), STARE (inter-anotador)     |
| Pérdidas evaluadas     | BCE, Dice, BCE+Dice combinada                        |
| Estrategias de dominio | Preprocesamiento CLAHE, fine-tuning                  |

---

## Estructura del repositorio

```
retinal_segmentation/
├── README.md                        ← este archivo
├── requirements.txt                 ← dependencias Python
├── run_all.sh                       ← pipeline completo (un solo comando)
├── colab_runner.ipynb               ← envoltorio para Google Colab
├── .gitignore
├── configs/                         ← configuraciones YAML por experimento
│   ├── train_attention_unet.yaml    ← experimento principal
│   ├── train_unet.yaml              ← ablación arquitectura
│   ├── ablation_bce.yaml            ← ablación pérdida BCE
│   ├── ablation_dice.yaml           ← ablación pérdida Dice
│   └── ablation_combined.yaml       ← ablación pérdida combinada
├── src/
│   ├── train.py                     ← script principal de entrenamiento
│   ├── data/
│   │   ├── dataset.py               ← DRIVE / STARE / CHASE_DB1
│   │   └── transforms.py            ← augmentación, CLAHE, normalización
│   ├── models/
│   │   ├── unet.py                  ← U-Net implementada desde cero
│   │   └── attention_unet.py        ← Attention U-Net con Attention Gates
│   └── utils/
│       ├── losses.py                ← BCE / Dice / Combined
│       ├── metrics.py               ← Sens, Spec, F1, AUC, IoU, Accuracy
│       ├── trainer.py               ← bucle de entrenamiento + early stopping
│       └── logger.py
├── scripts/
│   ├── setup_datasets.py            ← descarga automática desde Kaggle
│   ├── validate_datasets.py         ← validación de estructura y máscaras antes de entrenar
│   ├── evaluate.py                  ← evaluación + ROC + análisis de fallos
│   ├── domain_adaptation.py         ← DRIVE→CHASE_DB1 + CLAHE + fine-tuning
│   ├── vessel_thickness_analysis.py ← sensibilidad por grosor (entregable #5)
│   ├── inter_annotator_stare.py     ← concordancia entre anotadores
│   ├── plot_training_curves.py      ← curvas de entrenamiento
│   └── visualize_attention.py       ← mapas de atención de los gates
├── data/                            ← datasets (se descargan automáticamente)
├── outputs/                         ← checkpoints, métricas e historial
├── results/                         ← figuras y reportes finales
└── reports/                         ← informe NeurIPS (PDF)
```

---

## Cómo ejecutar — un solo comando

```bash
bash run_all.sh
```

Esto descarga los 3 datasets automáticamente desde Kaggle, entrena los 5 experimentos, ejecuta todos los análisis y genera el resumen final. Tiempo total: ~3 h en GPU L4, ~5-6 h en T4, ~10 min adicionales de descarga inicial.

---

## Configuración del entorno

El comando único requiere:

### 1. Python 3.10+ y dependencias

```bash
pip install -r requirements.txt
```

### 2. GPU NVIDIA con drivers CUDA (recomendado)

CPU funciona pero es ~30× más lento. Para Colab, ver siguiente sección.

### 3. Token de Kaggle (para descarga automática de datasets)

Los datasets DRIVE, STARE y CHASE_DB1 se descargan desde mirrors públicos en Kaggle. Necesitas un token de API gratuito.

Kaggle ofrece dos sistemas de autenticación; el script `setup_datasets.py` detecta automáticamente cualquiera de los dos:

#### Opción A — Sistema nuevo (recomendado por Kaggle)

1. Crea cuenta gratuita en <https://www.kaggle.com/account/login>
2. Ve a <https://www.kaggle.com/settings/api>
3. Click **Generate New Token** (sección "API Tokens (Recommended)")
4. Copia el token completo (un string que empieza con `KGAT_...`)
5. Guárdalo:

```bash
mkdir -p ~/.kaggle
echo 'KGAT_tu_token_aqui' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Alternativa: exportarlo como variable de entorno (útil para CI/CD):

```bash
export KAGGLE_API_TOKEN='KGAT_tu_token_aqui'
```

#### Opción B — Sistema legacy (`kaggle.json`)

Sigue funcionando. Ve a la misma página y bajo **"Legacy API Credentials"** click **Create Legacy API Key**. Descarga `kaggle.json` (contiene `{"username": ..., "key": ...}`).

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Mirrors usados

El script `setup_datasets.py` descarga de estos mirrors públicos:

| Dataset    | Slug Kaggle                                                       | Licencia        |
|------------|-------------------------------------------------------------------|-----------------|
| DRIVE      | `umairinayat/retinal-vessel-segmentation-datasets`                | Académica       |
| STARE      | `umairinayat/retinal-vessel-segmentation-datasets`                | Dominio público |
| CHASE_DB1  | `khoongweihao/chasedb1`                                           | CC BY 4.0       |

---

## Ejecución en Google Colab

El repo incluye `colab_runner.ipynb` para ejecutar todo en GPU sin instalar nada localmente. Usa el badge **"Open In Colab"** al inicio de este README, o:

1. Ve a <https://colab.research.google.com/> → **File → Open notebook → GitHub**.
2. Pega la URL de tu repo y selecciona `colab_runner.ipynb`.
3. **Entorno de ejecución → Cambiar tipo → GPU L4** (o T4 si no hay L4).
4. Ejecuta la celda 1: pega tu token de Kaggle (formato `KGAT_...`) cuando te lo pida. El input es invisible (getpass) por seguridad.
5. Edita `REPO_URL` en la celda 2 y ejecútala.

La celda 2 hace todo el trabajo: clona el repo, instala dependencias, descarga datasets, entrena y evalúa. No requiere intervención manual.

### ¿Qué GPU elegir en Colab Pro?

Para este proyecto (Attention U-Net, batch 4, 512×512, 100 épocas, ~33M parámetros):

| GPU  | VRAM  | CU/h  | Tiempo total | Recomendación |
|------|-------|-------|--------------|---------------|
| **T4**  | 16 GB | 1.96  | 5-6 h (~11 CU)  | Default seguro |
| **L4**  | 24 GB | 4-5   | 2-3 h (~12 CU)  | **Mejor relación tiempo/CU** |
| A100 | 40 GB | 10-15 | 1.5-2 h (~25 CU) | Solo si tienes prisa |
| H100 | 80 GB | premium | ~1 h         | Sobredimensionada |
| TPU  | —     | 1.76  | N/A          | **NO usar** (hooks + Attention Gates no son XLA-friendly) |

No actives "RAM amplia" — la RAM estándar (12 GB) sobra y ahorras CU.

---

## Estructura de datasets

Los datasets se descargan automáticamente al ejecutar `run_all.sh` y se organizan en `data/` con esta estructura:

```
data/
├── drive/
│   ├── training/{images, 1st_manual, mask}/   20 archivos c/u
│   └── test/{images, 1st_manual, mask}/        20 archivos c/u
├── chase_db1/
│   ├── images/                                  28 imágenes .jpg
│   └── labels/                                  28 máscaras 1stHO .png
└── stare/
    ├── images/                                  20 imágenes .ppm
    ├── labels-ah/                               20 anotador Adam Hoover
    └── labels-vk/                               20 anotador Valentina Kouznetsova
```

**Descarga manual** (si Kaggle no está disponible): puedes colocar los archivos manualmente en estas rutas. Fuentes oficiales para cada uno:
- DRIVE: <https://drive.grand-challenge.org/> (requiere registro)
- CHASE_DB1: <https://researchdata.kingston.ac.uk/96/> (CC BY 4.0, sin registro)
- STARE: <https://cecas.clemson.edu/~ahoover/stare/> (dominio público)

Citación CHASE_DB1: Fraz, M.M. et al., "An Ensemble Classification-Based Approach Applied to Retinal Blood Vessel Segmentation", IEEE TBME 2012.

---

## Ejecución por partes (avanzado)

Para iterar en componentes individuales después de tener el modelo entrenado:

```bash
# Solo entrenar el modelo principal
python src/train.py --config configs/train_attention_unet.yaml

# Solo evaluar (requiere checkpoint previo)
python scripts/evaluate.py \
    --config configs/train_attention_unet.yaml \
    --checkpoint outputs/attention_unet_drive/best_model.pt \
    --out_dir results/eval

# Solo análisis por grosor
python scripts/vessel_thickness_analysis.py \
    --config configs/train_attention_unet.yaml \
    --checkpoint outputs/attention_unet_drive/best_model.pt \
    --out_dir results/vessel_thickness

# Solo experimento de dominio
python scripts/domain_adaptation.py \
    --config configs/train_attention_unet.yaml \
    --checkpoint outputs/attention_unet_drive/best_model.pt \
    --chase_root data/chase_db1 \
    --out_dir results/domain_adaptation
```

---

## Reproducibilidad

- Todas las semillas (`random`, `numpy`, `torch`, `cuda`) se fijan en 42.
- `torch.backends.cudnn.deterministic = True`.
- Los splits son fijos (DRIVE) o reproducibles vía `torch.Generator().manual_seed(42)` (CHASE_DB1).

---

## Métricas implementadas

`src/utils/metrics.py` calcula:

- **Sensibilidad** (Recall / TPR) — fracción de vasos detectados
- **Especificidad** (TNR) — fracción de fondo correctamente clasificado
- **F1 (Dice)** — balance precisión/recall, métrica primaria
- **AUC-ROC** — capacidad discriminativa global (sobre probabilidades)
- **IoU (Jaccard)** — solapamiento
- **Accuracy** — precisión píxel a píxel

---

## Licencia y datasets

Este código se publica bajo MIT License. Los datasets tienen sus propias licencias:

- DRIVE: uso académico/no-comercial, requiere registro
- CHASE_DB1: CC BY 4.0
- STARE: dominio público (uso académico)

---

## Referencias principales

1. Ronneberger, O., Fischer, P., Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
2. Oktay, O. et al. *Attention U-Net: Learning Where to Look for the Pancreas*. MIDL 2018.
3. Staal, J. et al. *Ridge-Based Vessel Segmentation in Color Images of the Retina*. IEEE TMI 2004 (DRIVE).
4. Fraz, M.M. et al. *An Ensemble Classification-Based Approach Applied to Retinal Blood Vessel Segmentation*. IEEE TBME 2012 (CHASE_DB1).
5. Hoover, A. et al. *Locating Blood Vessels in Retinal Images by Piecewise Threshold Probing of a Matched Filter Response*. IEEE TMI 2000 (STARE).
6. Milletari, F., Navab, N., Ahmadi, S.-A. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV 2016 (Dice loss).