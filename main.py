"""
main.py
-------
Entrypoint per avviare l'addestramento del modello U-Net sulla segmentazione
dei vasi sanguigni nelle immagini Fundus.

Esecuzione:
    python main.py

Il training è configurabile interamente da src/config.py.
Al termine di ogni epoch, le metriche vengono stampate a console.
Il modello con il miglior Dice Score di validazione viene salvato in:
    output/models/best_unet_vessel.pth
"""

import random
import time
import numpy as np
import torch
from torch import optim

from src.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    PATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    BEST_MODEL_PATH,
    OUTPUT_DIR,
    UNET_FEATURES,
    create_directories,
    get_kaggle_dataset_path,
)
from src.dataset import get_dataloaders
from src.models.unet import UNet
from src.losses import CombinedLoss
from src.engine import train_one_epoch, validate, save_checkpoint
from src.logger import RunLogger


# ==============================================================================
# RIPRODUCIBILITÀ
# ==============================================================================

def set_seed(seed: int) -> None:
    """Fissa i seed per garantire riproducibilità degli esperimenti."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # --- Setup iniziale ---
    set_seed(RANDOM_SEED)
    create_directories()

    print("=" * 60)
    print("  VESSEL SEGMENTATION - TRAINING U-Net")
    print("=" * 60)
    print(f"  Device   : {DEVICE.upper()}")
    print(f"  Epochs   : {EPOCHS}")
    print(f"  LR       : {LEARNING_RATE}")
    print(f"  Batch    : {BATCH_SIZE}  |  Patch: {PATCH_SIZE}x{PATCH_SIZE}")
    print("=" * 60)

    # --- Dataset e DataLoader ---
    dataset_root = get_kaggle_dataset_path()
    train_loader, val_loader, _ = get_dataloaders(
        dataset_root=dataset_root,
        patch_size=PATCH_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # --- Modello, Loss, Ottimizzatore ---
    model   = UNet(in_channels=3, out_channels=1, features=UNET_FEATURES).to(DEVICE)
    loss_fn = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler: riduce il LR se la val_loss non migliora per 5 epoche
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # GradScaler per AMP (solo su GPU; su CPU viene ignorato in engine.py)
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

    # --- Logger ---
    logger = RunLogger(
        log_dir=OUTPUT_DIR / "logs",
        hparams={
            "patch_size":         PATCH_SIZE,
            "batch_size":         BATCH_SIZE,
            "epochs":             EPOCHS,
            "learning_rate":      LEARNING_RATE,
            "num_workers":        NUM_WORKERS,
            "device":             DEVICE,
            "random_seed":        RANDOM_SEED,
            "unet_features":      UNET_FEATURES,
            "loss_alpha":         0.5,
            "scheduler_patience": 5,
            "scheduler_factor":   0.5,
        },
    )
    logger.start()

    print(f"\n  Parametri U-Net: {model.count_parameters():,}\n")

    # --- Loop di Training ---
    best_val_dice  = 0.0
    best_val_epoch = 1

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch [{epoch:>3}/{EPOCHS}]")

        epoch_start = time.perf_counter()

        train_loss, train_dice, train_iou, batch_times = train_one_epoch(
            model, train_loader, optimizer, loss_fn, DEVICE, scaler
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, loss_fn, DEVICE
        )

        epoch_time = time.perf_counter() - epoch_start

        scheduler.step(val_loss)

        # Logging a console
        print(
            f"  Train -> Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}\n"
            f"  Val   -> Loss: {val_loss:.4f}   | Dice: {val_dice:.4f}   | IoU: {val_iou:.4f}"
        )

        # Logging su file
        logger.log_epoch(
            epoch, train_loss, train_dice, train_iou,
            val_loss, val_dice, val_iou,
            epoch_time, batch_times,
        )

        # Salvataggio del miglior modello
        if val_dice > best_val_dice:
            best_val_dice  = val_dice
            best_val_epoch = epoch
            save_checkpoint(model, BEST_MODEL_PATH, epoch, val_dice)

        print()

    logger.finish(best_val_dice, best_val_epoch)

    print("=" * 60)
    print(f"  Training completato. Miglior Dice Val: {best_val_dice:.4f}")
    print(f"  Modello salvato in: {BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
