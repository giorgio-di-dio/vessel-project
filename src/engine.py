"""
src/engine.py
-------------
Motore di training e validazione per la segmentazione dei vasi sanguigni.

Contiene:
    - train_one_epoch() : Un'intera epoch di forward + backward + optimizer step.
    - validate()        : Loop di validazione senza aggiornamento dei pesi.
    - save_checkpoint() : Salva i pesi del modello con le performance migliori.
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import dice_score, iou_score
from src.config import DEVICE


# ==============================================================================
# TRAINING
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str = DEVICE,
    scaler: torch.cuda.amp.GradScaler = None,
) -> Tuple[float, float, float]:
    """
    Esegue un'intera epoch di training: forward, backward e aggiornamento pesi.

    Supporto opzionale per Automatic Mixed Precision (AMP):
        Se 'scaler' è fornito (non None), utilizza torch.cuda.amp per
        velocizzare il training su GPU con precisione float16 dove possibile,
        riducendo l'utilizzo della VRAM fino al 50%.
        Su CPU il parametro viene ignorato.

    Args:
        model     : Il modello U-Net.
        loader    : DataLoader del set di training.
        optimizer : Ottimizzatore (es. Adam).
        loss_fn   : Funzione di loss (es. CombinedLoss).
        device    : 'cuda' o 'cpu'.
        scaler    : GradScaler per AMP (None per disabilitare).

    Returns:
        Tuple (avg_loss, avg_dice, avg_iou) medie sull'intera epoch.
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    loop = tqdm(loader, desc="  [Train]", leave=False, unit="batch")

    for images, masks in loop:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        if scaler is not None and device == "cuda":
            # --- Automatic Mixed Precision (solo su GPU) ---
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss = loss_fn(predictions, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # --- Training standard (CPU o GPU senza AMP) ---
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            loss.backward()
            optimizer.step()

        # Calcolo metriche per il logging (detach dalla computational graph)
        batch_loss = loss.item()
        batch_dice = dice_score(predictions.detach(), masks, from_logits=True)
        batch_iou  = iou_score(predictions.detach(), masks, from_logits=True)

        total_loss += batch_loss
        total_dice += batch_dice
        total_iou  += batch_iou

        loop.set_postfix(loss=f"{batch_loss:.4f}", dice=f"{batch_dice:.4f}")

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ==============================================================================
# VALIDAZIONE
# ==============================================================================

def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str = DEVICE,
) -> Tuple[float, float, float]:
    """
    Valuta le performance del modello sul set di validazione senza aggiornare i pesi.

    Utilizza torch.no_grad() per disabilitare il calcolo dei gradienti,
    riducendo il consumo di memoria e velocizzando l'inferenza.

    Args:
        model   : Il modello U-Net (in modalità eval).
        loader  : DataLoader del set di validazione.
        loss_fn : Funzione di loss per il monitoraggio.
        device  : 'cuda' o 'cpu'.

    Returns:
        Tuple (avg_loss, avg_dice, avg_iou) medie sull'intero set di validazione.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    loop = tqdm(loader, desc="  [Val]  ", leave=False, unit="img")

    with torch.no_grad():
        for images, masks in loop:
            images = images.to(device)
            masks  = masks.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, masks)

            total_loss += loss.item()
            total_dice += dice_score(predictions, masks, from_logits=True)
            total_iou  += iou_score(predictions, masks, from_logits=True)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ==============================================================================
# CHECKPOINT
# ==============================================================================

def save_checkpoint(model: nn.Module, path: Path, epoch: int, val_dice: float) -> None:
    """
    Salva i pesi del modello e i metadati di training in un file .pth.

    Il checkpoint contiene:
        - 'state_dict': Pesi del modello.
        - 'epoch'     : Epoch in cui è avvenuto il salvataggio.
        - 'val_dice'  : Dice Score di validazione che ha motivato il salvataggio.

    Args:
        model    : Il modello U-Net da salvare.
        path     : Path completo del file .pth.
        epoch    : Numero dell'epoch corrente.
        val_dice : Valore del Dice Score di validazione.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_dice": val_dice,
            "state_dict": model.state_dict(),
        },
        path,
    )
    print(f"  [Checkpoint] Salvato: {path.name} (epoch={epoch}, dice={val_dice:.4f})")


def load_checkpoint(model: nn.Module, path: Path, device: str = DEVICE) -> dict:
    """
    Carica i pesi da un checkpoint .pth nel modello.

    Args:
        model  : Istanza del modello (stessa architettura usata durante il saving).
        path   : Path al file .pth.
        device : Device su cui mappare i tensori.

    Returns:
        Il dizionario del checkpoint (con 'epoch' e 'val_dice').
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print(
        f"  [Checkpoint] Caricato: {Path(path).name} "
        f"(epoch={checkpoint['epoch']}, val_dice={checkpoint['val_dice']:.4f})"
    )
    return checkpoint
