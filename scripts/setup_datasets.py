"""
scripts/setup_datasets.py — Descarga y organiza DRIVE, STARE y CHASE_DB1 desde Kaggle.

Llamado por `run_all.sh` en el paso 1. También se puede correr standalone:

  python scripts/setup_datasets.py            # descarga + organiza los 3
  python scripts/setup_datasets.py --skip-download   # solo organiza (si ya descargaste)
  python scripts/setup_datasets.py --only drive      # solo uno

Requisitos:
  - kaggle CLI instalado (pip install kaggle)
  - ~/.kaggle/kaggle.json con permisos 600 (chmod 600 ~/.kaggle/kaggle.json)

Estrategia: cada mirror de Kaggle empaqueta los archivos de forma distinta.
En vez de asumir una estructura, este script camina los archivos descomprimidos,
los identifica por patrón de nombre, y los mueve a la estructura esperada:

  data/drive/training/{images, 1st_manual, mask}/
  data/drive/test/{images, 1st_manual, mask}/
  data/chase_db1/{images, labels}/
  data/stare/{images, labels-ah, labels-vk}/
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# ── Mirrors de Kaggle ─────────────────────────────────────────────────────
# Decisiones:
#   - DRIVE y STARE comparten el mismo mirror combinado (umairinayat) porque
#     el mirror estándar de DRIVE (andrewmvd) NO incluye las anotaciones
#     manuales del test set, y el de STARE (vidheeshnacode) solo tiene
#     imágenes crudas sin labels AH/VK. El combinado pesa más (~1.5 GB)
#     pero es la única fuente confiable que tiene todos los archivos.
#   - CHASE_DB1 sigue con su mirror dedicado porque es pequeño (~2 MB) y
#     funciona bien.
#   - El script detecta automáticamente que DRIVE y STARE comparten slug
#     y solo descarga una vez (ver download_unique_sources).
KAGGLE_SLUGS = {
    "drive":     "umairinayat/retinal-vessel-segmentation-datasets",
    "stare":     "umairinayat/retinal-vessel-segmentation-datasets",
    "chase_db1": "khoongweihao/chasedb1",
}


# ── Utilidades ────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    prefix = {"INFO": "  ", "OK": "  ✓ ", "WARN": "  ⚠ ", "ERR": "  ✗ "}[level]
    print(f"{prefix}{msg}")


def check_kaggle_credentials():
    """
    Verifica que haya credenciales de Kaggle disponibles. Soporta los 2 sistemas
    actuales (ambos siguen funcionando con el CLI ≥ 1.8):

      NUEVO (recomendado por Kaggle):
        - Variable de entorno KAGGLE_API_TOKEN con el token KGAT_...
        - O archivo ~/.kaggle/access_token con el token KGAT_...

      LEGACY (sigue soportado):
        - Archivo ~/.kaggle/kaggle.json con {"username": "...", "key": "..."}
        - O variables de entorno KAGGLE_USERNAME + KAGGLE_KEY

    Devuelve True si encuentra cualquiera, False si ninguna está presente.
    """
    home = Path.home()

    # ── Sistema nuevo: KAGGLE_API_TOKEN ─────────────────────────────
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        log("Credenciales: variable de entorno KAGGLE_API_TOKEN", "OK")
        return True

    access_token = home / ".kaggle" / "access_token"
    if access_token.exists() and access_token.stat().st_size > 0:
        log(f"Credenciales: {access_token}", "OK")
        if os.name == "posix":
            mode = access_token.stat().st_mode & 0o777
            if mode != 0o600:
                log(f"Corrigiendo permisos de {access_token} a 600...", "WARN")
                os.chmod(access_token, 0o600)
        return True

    # ── Sistema legacy: kaggle.json ─────────────────────────────────
    kaggle_json = home / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        log(f"Credenciales: {kaggle_json} (legacy)", "OK")
        if os.name == "posix":
            mode = kaggle_json.stat().st_mode & 0o777
            if mode != 0o600:
                log(f"Corrigiendo permisos de {kaggle_json} a 600...", "WARN")
                os.chmod(kaggle_json, 0o600)
        return True

    # ── Legacy via env vars (kagglehub-style) ───────────────────────
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        log("Credenciales: KAGGLE_USERNAME + KAGGLE_KEY (env)", "OK")
        return True

    # ── No hay credenciales: instrucciones claras para ambos sistemas ─
    log("NO se encontraron credenciales de Kaggle", "ERR")
    log("", "INFO")
    log("Tienes 2 opciones para configurarlas:", "INFO")
    log("", "INFO")
    log("OPCIÓN A (recomendada, sistema nuevo de Kaggle):", "INFO")
    log("  1. Ve a https://www.kaggle.com/settings/api", "INFO")
    log("  2. Click 'Generate New Token' → copia el token KGAT_...", "INFO")
    log("  3. Guárdalo así (reemplaza KGAT_... con el tuyo):", "INFO")
    log("     mkdir -p ~/.kaggle", "INFO")
    log("     echo 'KGAT_...' > ~/.kaggle/access_token", "INFO")
    log("     chmod 600 ~/.kaggle/access_token", "INFO")
    log("", "INFO")
    log("OPCIÓN B (legacy, sigue funcionando):", "INFO")
    log("  1. En https://www.kaggle.com/settings/api → sección 'Legacy API Credentials'", "INFO")
    log("  2. Click 'Create Legacy API Key' → descarga kaggle.json", "INFO")
    log("  3. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json", "INFO")
    return False


def ensure_kaggle_installed():
    """Instala el CLI de kaggle si no está."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("Instalando kaggle CLI...", "INFO")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "kaggle"],
            check=True
        )
        return True


def kaggle_download(slug: str, out_dir: Path) -> Path:
    """
    Descarga un dataset de Kaggle al directorio out_dir.
    Devuelve la ruta del primer .zip encontrado.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Descargando {slug}...", "INFO")
    res = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        log(f"Falló descarga de {slug}", "ERR")
        log(res.stderr.strip(), "ERR")
        raise RuntimeError(f"kaggle download falló para {slug}")

    zips = list(out_dir.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"No se encontró .zip tras descarga de {slug}")
    return zips[0]


def safe_extract(zip_path: Path, extract_to: Path):
    """Descomprime un zip, ignorando rutas peligrosas (zip slip)."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = extract_to / member
            # Defensa básica zip-slip
            if not str(target.resolve()).startswith(str(extract_to.resolve())):
                continue
            zf.extract(member, extract_to)
    log(f"Descomprimido en {extract_to}", "OK")


def find_files(root: Path, patterns):
    """Busca archivos que coincidan con cualquiera de los patrones (regex sobre nombre)."""
    matches = []
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(c.search(path.name) for c in compiled):
            matches.append(path)
    return matches


def move_files(files, target_dir: Path):
    """Mueve archivos al directorio destino (creándolo si hace falta)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = target_dir / f.name
        if dst.exists():
            continue
        shutil.move(str(f), str(dst))
    return len(files)


# ── Organizadores específicos por dataset ─────────────────────────────────
def organize_drive(extracted: Path, target_root: Path):
    """
    Organiza los archivos DRIVE usando patrones de nombre estrictos.
    DRIVE tiene nomenclatura única:
      - Imágenes training: ##_training.tif  (21-40)
      - Imágenes test:     ##_test.tif      (01-20)
      - Labels training:   ##_manual1.gif   (21-40)
      - Labels test:       ##_manual1.gif   (01-20)
      - Mask training:     ##_training_mask.gif
      - Mask test:         ##_test_mask.gif
    Usar patrones estrictos evita capturar archivos de otros datasets
    (HRF, FIVES, CHASE) presentes en el zip combinado de umairinayat.
    """
    log("Organizando DRIVE...", "INFO")

    def number_of(f: Path) -> int:
        m = re.match(r"(\d+)", f.name)
        return int(m.group(1)) if m else -1

    # Imágenes — solo extensión .tif con sufijo _training o _test
    train_imgs = find_files(extracted, [r"^\d+_training\.tif$"])
    test_imgs  = find_files(extracted, [r"^\d+_test\.tif$"])

    # Labels manuales — .gif con _manual1
    all_man   = find_files(extracted, [r"^\d+_manual1\.gif$"])
    train_man = [f for f in all_man if number_of(f) >= 21]
    test_man  = [f for f in all_man if number_of(f) <= 20]

    # Máscaras FOV
    train_fov = find_files(extracted, [r"^\d+_training_mask\.gif$"])
    test_fov  = find_files(extracted, [r"^\d+_test_mask\.gif$"])

    counts = {
        "training/images":     move_files(train_imgs, target_root / "training/images"),
        "training/1st_manual": move_files(train_man,  target_root / "training/1st_manual"),
        "training/mask":       move_files(train_fov,  target_root / "training/mask"),
        "test/images":         move_files(test_imgs,  target_root / "test/images"),
        "test/1st_manual":     move_files(test_man,   target_root / "test/1st_manual"),
        "test/mask":           move_files(test_fov,   target_root / "test/mask"),
    }

    for k, v in counts.items():
        log(f"{target_root}/{k}: {v} archivos", "OK")

    missing = {k: 20 - counts.get(k, 0)
               for k in counts if counts[k] < 20}
    if missing:
        log(f"DRIVE incompleto, faltan: {missing}", "WARN")
        if "test/1st_manual" in missing:
            log("Este mirror no incluye labels de test (limitación conocida).", "WARN")
            log("El entrenamiento funcionará; evaluación en test usará val split.", "INFO")


def organize_chase_db1(extracted: Path, target_root: Path):
    """
    CHASE_DB1 espera:
      data/chase_db1/images/Image_01L.jpg ... Image_14R.jpg
      data/chase_db1/labels/Image_01L_1stHO.png ... (usamos solo 1stHO)

    El mirror khoongweihao trae todo plano: Image_XXY.jpg, Image_XXY_1stHO.png,
    Image_XXY_2ndHO.png.
    """
    log("Organizando CHASE_DB1...", "INFO")

    images = find_files(extracted, [r"^Image_\d+[LR]\.jpg$", r"^Image_\d+[LR]\.png$"])
    labels = find_files(extracted, [r"_1stHO\.png$"])
    # Nota: ignoramos _2ndHO.png deliberadamente. Si se quisiera usar como
    # segundo anotador, se podría guardar en labels-2nd/.

    n_img = move_files(images, target_root / "images")
    n_lab = move_files(labels, target_root / "labels")

    log(f"{target_root}/images: {n_img} archivos", "OK")
    log(f"{target_root}/labels: {n_lab} archivos", "OK")

    if n_img != 28 or n_lab != 28:
        log(f"CHASE_DB1 esperaba 28+28 archivos, encontrado {n_img}+{n_lab}", "WARN")


def organize_stare(extracted: Path, target_root: Path):
    """
    STARE espera:
      data/stare/images/im0001.ppm ...
      data/stare/labels-ah/im0001.ah.ppm ...
      data/stare/labels-vk/im0001.vk.ppm ...

    El mirror combinado umairinayat trae STARE dentro de un subdir
    (típicamente STARE/ o stare-data/) junto con otros datasets.
    Este organizador localiza los archivos STARE específicamente
    y los mueve. También soporta archivos comprimidos (.gz).
    """
    log("Organizando STARE...", "INFO")

    # Buscar archivos de anotaciones (patrón claro: .ah.ppm o .vk.ppm,
    # con o sin .gz). Soportamos ambas extensiones.
    ah = []
    vk = []
    for ext in [".ppm", ".ppm.gz", ".png", ".jpg", ".gif", ".tif"]:
        ah.extend(extracted.rglob(f"*.ah{ext}"))
        ah.extend(extracted.rglob(f"*_ah{ext}"))
        vk.extend(extracted.rglob(f"*.vk{ext}"))
        vk.extend(extracted.rglob(f"*_vk{ext}"))

    # Imágenes STARE: archivos con patrón im####.ppm (im + dígitos)
    # Usamos regex estricto para no agarrar files como "image_4.png" del ruido.
    label_names = {f.name for f in ah} | {f.name for f in vk}
    img_patt = re.compile(
        r"^im\d+\.(ppm|ppm\.gz|png|jpg|jpeg|tif|tiff)$",
        re.IGNORECASE
    )
    images = []
    for f in extracted.rglob("*"):
        if not f.is_file():
            continue
        if not img_patt.match(f.name):
            continue
        if f.name in label_names:
            continue
        images.append(f)

    # Mover archivos
    def safe_move(files, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for f in files:
            dst = target_dir / f.name
            if not dst.exists():
                shutil.move(str(f), str(dst))
                moved += 1
        return moved

    n_img = safe_move(images, target_root / "images")
    n_ah  = safe_move(ah,     target_root / "labels-ah")
    n_vk  = safe_move(vk,     target_root / "labels-vk")

    log(f"{target_root}/images:    {n_img} archivos", "OK")
    log(f"{target_root}/labels-ah: {n_ah} archivos", "OK")
    log(f"{target_root}/labels-vk: {n_vk} archivos", "OK")

    if n_ah == 0 or n_vk == 0:
        log("STARE sin anotaciones AH/VK. El análisis inter-anotador se omitirá.", "WARN")
        log("(El resto del pipeline funcionará normal — STARE es solo bonus.)", "INFO")
    elif n_img < 20:
        log(f"STARE: solo {n_img} imágenes (esperado ≥20)", "WARN")


# ── Detector de "ya está descargado" ──────────────────────────────────────
def already_organized(name: str, target_root: Path) -> bool:
    """Heurística: si el directorio principal ya tiene archivos, asumimos OK."""
    if not target_root.exists():
        return False
    if name == "drive":
        p = target_root / "training" / "images"
        return p.exists() and len(list(p.glob("*.tif"))) >= 15
    if name == "chase_db1":
        p = target_root / "images"
        return p.exists() and len(list(p.glob("*.jpg"))) >= 25
    if name == "stare":
        p = target_root / "images"
        return p.exists() and len(list(p.glob("*.ppm"))) >= 18
    return False


# ── Punto de entrada ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data",
                        help="Directorio base donde se colocarán los datasets")
    parser.add_argument("--only", type=str, default=None,
                        choices=list(KAGGLE_SLUGS.keys()),
                        help="Procesar solo este dataset")
    parser.add_argument("--skip-download", action="store_true",
                        help="No descargar de Kaggle, solo reorganizar archivos en data_root")
    parser.add_argument("--force", action="store_true",
                        help="Re-descargar incluso si ya está organizado")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    targets = {
        "drive":     data_root / "drive",
        "stare":     data_root / "stare",
        "chase_db1": data_root / "chase_db1",
    }
    organizers = {
        "drive":     organize_drive,
        "stare":     organize_stare,
        "chase_db1": organize_chase_db1,
    }

    names = [args.only] if args.only else list(KAGGLE_SLUGS.keys())

    # Verificaciones previas (solo si vamos a descargar)
    if not args.skip_download:
        if not check_kaggle_credentials():
            sys.exit(1)
        ensure_kaggle_installed()

    # Filtrar datasets que ya están organizados (a menos que --force)
    pending = []
    for name in names:
        if not args.force and already_organized(name, targets[name]):
            log(f"{name} ya organizado en {targets[name]}, saltando", "OK")
            continue
        pending.append(name)

    if not pending:
        print("\n✓ Todos los datasets ya están organizados.")
        return

    # Agrupar por slug para descargar cada uno SOLO UNA VEZ
    # (ej. DRIVE y STARE comparten el slug de umairinayat)
    by_slug = {}
    for name in pending:
        slug = KAGGLE_SLUGS[name]
        by_slug.setdefault(slug, []).append(name)

    # Descargar cada slug único y correr todos los organizadores que lo necesiten
    for slug, dataset_names in by_slug.items():
        print(f"\n=== {slug} ({', '.join(dataset_names)}) ===")

        if args.skip_download:
            log(f"skip_download=True, esperando archivos en {data_root}", "INFO")
            extracted_dir = data_root
            for name in dataset_names:
                print(f"\n  → Organizando {name.upper()}...")
                organizers[name](extracted_dir, targets[name])
            continue

        with tempfile.TemporaryDirectory(prefix=f"{slug.replace('/','_')}_") as tmpdir:
            tmpdir = Path(tmpdir)
            try:
                zip_path = kaggle_download(slug, tmpdir)
            except Exception as e:
                log(f"Error descargando {slug}: {e}", "ERR")
                sys.exit(1)
            safe_extract(zip_path, tmpdir / "extracted")

            # Correr todos los organizadores que usan este slug
            for name in dataset_names:
                print(f"\n  → Organizando {name.upper()}...")
                organizers[name](tmpdir / "extracted", targets[name])

    print("\n✓ Setup de datasets completado.")


if __name__ == "__main__":
    main()
