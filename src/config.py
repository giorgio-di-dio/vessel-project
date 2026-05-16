import os
import torch
import kagglehub
from pathlib import Path

# ==============================================================================
# 1. PERCORSI E DIRECTORY PRINCIPALI
# ==============================================================================
# Otteniamo la directory radice del progetto (due livelli sopra questo file)
BASE_DIR = Path(__file__).resolve().parent.parent

# ID del Dataset su Kaggle
KAGGLE_DATASET_ID = "nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation"

# Cartelle per i dati
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

def get_kaggle_dataset_path() -> Path:
    """
    Scarica (se necessario) e restituisce il percorso locale della cache
    di kagglehub contenente le immagini e le maschere del dataset.
    """
    print(f"[Kaggle] Richiesta/Verifica dataset: {KAGGLE_DATASET_ID}")
    path_str = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    return Path(path_str)

# Cartelle per gli output (modelli salvati, log, predizioni visive)
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# ==============================================================================
# 2. IPERPARAMETRI DI ADDESTRAMENTO
# ==============================================================================
# Dimensioni dell'immagine di input per la CNN (es. U-Net)
# Immagini di dimensioni standard come 256x256 o 512x512 sono ideali per la segmentazione
IMAGE_HEIGHT = 2048
IMAGE_WIDTH = 2048
CHANNELS = 3  # RGB

# Iperparametri di training
PATCH_SIZE = 512       # Patch size per il training (deve essere multiplo di 32) (256 con rete "piccola")
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 0        # 0 = sincrono (più stabile su Windows; aumentare su Linux/Mac)

# Configurazione del seed per la riproducibilità
RANDOM_SEED = 42

# ==============================================================================
# 3. CONFIGURAZIONE HARDWARE (DEVICE)
# ==============================================================================
# Rilevamento automatico della GPU se disponibile, altrimenti fallback su CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 4. IMPOSTAZIONI DEL MODELLO E CHECKPOINT
# ==============================================================================
# ?? Percorso in cui salvare i pesi del modello con le performance migliori
BEST_MODEL_PATH = MODELS_DIR / "best_unet_vessel.pth"

# Soglia di confidenza per binarizzare le probabilità della maschera predetta
MASK_THRESHOLD = 0.5


def create_directories():
    """
    Crea tutte le directory necessarie per il progetto se non esistono.
    """
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[Config] Verificata/Creata directory: {d}")

if __name__ == "__main__":
    # ?? Test di verifica rapida della configurazione
    print("=== VESSEL SEGMENTATION CONFIGURATION ===")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Hardware Device: {DEVICE.upper()}")
    create_directories()
    
    # ?? Verifica e ottieni il path di Kagglehub
    dataset_path = get_kaggle_dataset_path()
    print(f"Path to dataset files: {dataset_path}")
