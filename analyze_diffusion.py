"""
analyze_diffusion.py
--------------------
Script di inferenza e analisi per il modello Diffusion.
Riprende la logica di analyze.py ma utilizza la Pipeline Diffusion per testare il modello
addestrato sul set di test e generare le metriche finali (Dice e IoU).

Esecuzione:
    python analyze_diffusion.py
"""

import time
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm

from src.config import (
    DEVICE,
    RESULTS_DIR,
    OUTPUT_DIR,
    create_directories,
    get_kaggle_dataset_path,
)
from src.metrics import dice_score, iou_score
from src.models.diffusion_scheduler import DiffusionScheduler
from src.models.diffusion_unet import ConditionalUNet
from src.models.diffusion_inference import DiffusionPipeline
from src.diffusion_logger import DiffusionLogger

# Importiamo la funzione di salvataggio dal tuo analizzatore standard!
from analyze import save_comparison_figure

def main():
    create_directories()

    print("=" * 60)
    print("  VESSEL SEGMENTATION - ANALISI DIFFUSION")
    print("=" * 60)

    # Assicurati che punti alla cartella dove hai salvato i pesi della diffusion
    # Se hai salvato su drive, metti qui il path di drive!
    CHECKPOINT_PATH = Path("output/models/diffusion_checkpoints/diffusion_best.pth")

    if not CHECKPOINT_PATH.exists():
        print(f"\n[ERRORE] Nessun checkpoint trovato in: {CHECKPOINT_PATH}")
        print("         Assicurati di aver finito il training o scaricato i pesi da Drive.")
        return

    # --- Caricamento del modello Diffusion ---
    print(f"Caricamento modello da: {CHECKPOINT_PATH}")
    scheduler = DiffusionScheduler(num_timesteps=1000, device=DEVICE)
    model = ConditionalUNet(image_channels=3, mask_channels=1, base_dim=32).to(DEVICE)
    
    # Carichiamo i pesi
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    # Inizializziamo la pipeline (che mette il modello in eval in automatico)
    pipeline = DiffusionPipeline(model=model, scheduler=scheduler, device=DEVICE)

    # --- Percorsi del set di test ---
    dataset_root  = get_kaggle_dataset_path()
    test_img_dir  = dataset_root / "test" / "Original"
    test_mask_dir = dataset_root / "test" / "Ground truth"

    image_paths = sorted(test_img_dir.glob("*.png"))
    if not image_paths:
        print(f"[ERRORE] Nessuna immagine trovata in: {test_img_dir}")
        return

    # --- DEBUG/TEST VELOCE ---
    # Limita il numero di immagini per non aspettare ore durante i test
    MAX_IMAGES = 2
    if len(image_paths) > MAX_IMAGES:
        print(f"\n  [TEST MODE] Limito l'analisi alle prime {MAX_IMAGES} immagini (su {len(image_paths)} totali).")
        image_paths = image_paths[:MAX_IMAGES]

    print(f"\n  Immagini da processare: {len(image_paths)}")
    print(f"  Risultati salvati in: {RESULTS_DIR}\n")

    # --- LOGGER ---
    logger = DiffusionLogger(
        log_dir=Path("output/logs_diffusion"),
        hparams={
            "patch_size": 256,
            "stride": 128,
            "num_timesteps": 1000,
            "base_dim": 32,
            "use_ddim": True,
            "device": DEVICE,
        }
    )
    logger.start()

    total_dice = 0.0
    total_iou  = 0.0
    processed  = 0

    for img_path in tqdm(image_paths, desc="  Inference", unit="img"):
        mask_path = test_mask_dir / img_path.name
        if not mask_path.exists():
            continue

        # Caricamento immagine (RGB) e maschera (Grayscale)
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_mask   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Conversione immagine in tensore PyTorch (1, 3, H, W)
        image_tensor = (image_rgb.astype(np.float32) / 255.0)
        image_tensor = np.transpose(image_tensor, (2, 0, 1))
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(DEVICE)

        # --- INFERENZA DIFFUSION ---
        img_start = time.perf_counter()
        # L'algoritmo di stitching gestirà automaticamente le patch e le ricucirà
        pred_tensor = pipeline.infer_full_image(
            image_tensor, 
            patch_size=256, # Stessa size usata nel training
            stride=128,     # Sovrapposizione del 50% per bordi morbidi
            use_ddim=False
        )
        img_time = time.perf_counter() - img_start
        
        # Mettiamo in numpy array 2D per il plot
        pred_map = pred_tensor.squeeze().cpu().numpy()

        # Calcolo metriche per questa immagine
        gt_tensor = torch.from_numpy((gt_mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)

        img_dice = dice_score(pred_tensor, gt_tensor, from_logits=False)
        img_iou  = iou_score(pred_tensor, gt_tensor, from_logits=False)

        total_dice += img_dice
        total_iou  += img_iou
        processed  += 1

        logger.log_image(
            image_name=img_path.name,
            dice=img_dice,
            iou=img_iou,
            inference_time_s=img_time
        )

        # Salvataggio figura comparativa riutilizzando il tuo metodo esistente
        out_path = RESULTS_DIR / f"diff_result_{img_path.stem}.png"
        save_comparison_figure(image_rgb, gt_mask, pred_map, out_path)

    if processed > 0:
        avg_dice = total_dice / processed
        avg_iou  = total_iou  / processed

        logger.finish(avg_dice, avg_iou)

        print(f"\n{'=' * 60}")
        print(f"  RISULTATI FINALI DIFFUSION ({processed} immagini di test)")
        print(f"{'=' * 60}")
        print(f"  Dice Score medio : {avg_dice:.4f}")
        print(f"  IoU medio        : {avg_iou:.4f}")
        print(f"{'=' * 60}")
    else:
        print("[ATTENZIONE] Nessuna immagine processata.")

if __name__ == "__main__":
    main()
