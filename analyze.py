"""
analyze.py
----------
Script di inferenza, analisi visiva e calcolo delle metriche finali sul set di test.

Funzionalità:
    1. Sliding Window Inference  : Predice la maschera su immagini 2048x2048
                                   usando una finestra mobile 512x512 con overlap.
    2. Visualizzazione comparativa: Salva affiancati immagine originale,
                                   maschera ground truth e predizione del modello.
    3. Metriche finali           : Calcola Dice Score e IoU medi sul set di test.

Esecuzione:
    python analyze.py

Prerequisiti:
    - Avere un checkpoint addestrato in output/models/best_unet_vessel.pth (lanciare prima main.py).
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")   # Backend non-interattivo per salvare su file senza display
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import (
    DEVICE,
    PATCH_SIZE,
    MASK_THRESHOLD,
    BEST_MODEL_PATH,
    RESULTS_DIR,
    create_directories,
    get_kaggle_dataset_path,
)
from src.models.unet import UNet
from src.engine import load_checkpoint
from src.metrics import dice_score, iou_score


# ==============================================================================
# SLIDING WINDOW INFERENCE
# ==============================================================================

def sliding_window_inference(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = 384,
    device: str = DEVICE,
) -> np.ndarray:
    """
    Predice la maschera di segmentazione per un'immagine ad alta risoluzione
    usando la tecnica della finestra mobile (Sliding Window Inference).

    Strategia:
        1. Scorrere l'immagine con una finestra 'patch_size x patch_size' e
           uno stride 'stride' (< patch_size per garantire overlap).
        2. Per ogni patch: forward pass del modello -> probabilità sigmoid.
        3. Accumulo delle predizioni nelle zone di overlap tramite media semplice.
        4. Normalizzazione finale con il contatore di accumulazioni per pixel.

    L'overlap (patch_size - stride = 128 pixel di default) mitiga gli artefatti
    ai bordi delle patch (i pixel centrali di ogni patch ricevono più contesto
    del perimetro, che è meno accurato).

    Args:
        model      : U-Net con pesi caricati, in modalità eval.
        image_rgb  : Immagine RGB (H, W, 3), uint8, [0-255].
        patch_size : Dimensione della finestra di inferenza (default: 512).
        stride     : Passo della finestra (default: 384). overlap = patch_size - stride.
        device     : 'cuda' o 'cpu'.

    Returns:
        prediction_map : Mappa di probabilità (H, W), float32, [0.0, 1.0].
    """
    H, W = image_rgb.shape[:2]
    model.eval()

    # Mappa di accumulazione delle predizioni e conteggio delle sovrapposizioni
    prediction_map = np.zeros((H, W), dtype=np.float32)
    count_map      = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for r in range(0, H - patch_size + 1, stride):
            for c in range(0, W - patch_size + 1, stride):
                # Estrai la patch
                patch = image_rgb[r:r + patch_size, c:c + patch_size]

                # Normalizza e converti in tensore (1, 3, patch_size, patch_size)
                patch_tensor = (patch.astype(np.float32) / 255.0)
                patch_tensor = np.transpose(patch_tensor, (2, 0, 1))
                patch_tensor = torch.from_numpy(patch_tensor).unsqueeze(0).to(device)

                # Forward pass -> logit -> probabilità
                logit = model(patch_tensor)
                prob  = torch.sigmoid(logit).squeeze().cpu().numpy()  # (patch_size, patch_size)

                # Accumulo nella mappa globale
                prediction_map[r:r + patch_size, c:c + patch_size] += prob
                count_map[r:r + patch_size, c:c + patch_size]       += 1.0

    # Normalizzazione: media tra le predizioni sovrapposte
    # Evita divisione per zero nelle zone non coperte (bordo estremo)
    count_map = np.maximum(count_map, 1.0)
    prediction_map /= count_map

    return prediction_map


# ==============================================================================
# VISUALIZZAZIONE
# ==============================================================================

def save_comparison_figure(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_map: np.ndarray,
    output_path: Path,
    threshold: float = MASK_THRESHOLD,
) -> None:
    """
    Salva un'immagine comparativa con 3 pannelli affiancati:
        [Immagine Originale] | [Ground Truth] | [Predizione Binarizzata]

    Args:
        image_rgb   : Immagine RGB originale (H, W, 3), uint8.
        gt_mask     : Maschera ground truth (H, W), uint8 [0-255].
        pred_map    : Mappa di probabilità predetta (H, W), float32 [0-1].
        output_path : Path del file .png da salvare.
        threshold   : Soglia per binarizzare pred_map.
    """
    # Binarizzazione della predizione
    pred_binary = (pred_map > threshold).astype(np.uint8) * 255
    gt_binary   = (gt_mask > 127).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")

    titles = ["Immagine Originale", "Ground Truth", "Predizione U-Net"]
    images = [image_rgb, gt_binary, pred_binary]
    cmaps  = [None, "gray", "gray"]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=13, pad=10)
        ax.axis("off")

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=80, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    create_directories()

    print("=" * 60)
    print("  VESSEL SEGMENTATION - ANALISI E INFERENZA")
    print("=" * 60)

    # --- Verifica del checkpoint ---
    if not BEST_MODEL_PATH.exists():
        print(f"\n[ERRORE] Nessun checkpoint trovato in: {BEST_MODEL_PATH}")
        print("         Avvia prima il training con: python main.py")
        return

    # --- Caricamento del modello ---
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(model, BEST_MODEL_PATH, device=DEVICE)
    model.eval()

    # --- Percorsi del set di test ---
    dataset_root  = get_kaggle_dataset_path()
    test_img_dir  = dataset_root / "test" / "Original"
    test_mask_dir = dataset_root / "test" / "Ground truth"

    image_paths = sorted(test_img_dir.glob("*.png"))
    if not image_paths:
        print(f"[ERRORE] Nessuna immagine trovata in: {test_img_dir}")
        return

    print(f"\n  Immagini di test trovate: {len(image_paths)}")
    print(f"  Risultati salvati in: {RESULTS_DIR}\n")

    total_dice = 0.0
    total_iou  = 0.0
    processed  = 0

    for img_path in tqdm(image_paths, desc="  Inference", unit="img"):
        mask_path = test_mask_dir / img_path.name
        if not mask_path.exists():
            continue

        # Caricamento immagine e maschera
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_mask   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Inferenza con sliding window
        pred_map = sliding_window_inference(
            model, image_rgb, patch_size=PATCH_SIZE, stride=384, device=DEVICE
        )

        # Calcolo metriche per questa immagine
        pred_tensor = torch.from_numpy(pred_map).unsqueeze(0).unsqueeze(0)
        gt_tensor   = torch.from_numpy((gt_mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)

        img_dice = dice_score(pred_tensor, gt_tensor, from_logits=False)
        img_iou  = iou_score(pred_tensor, gt_tensor, from_logits=False)

        total_dice += img_dice
        total_iou  += img_iou
        processed  += 1

        # Salvataggio figura comparativa
        out_path = RESULTS_DIR / f"result_{img_path.stem}.png"
        save_comparison_figure(image_rgb, gt_mask, pred_map, out_path)

    if processed > 0:
        avg_dice = total_dice / processed
        avg_iou  = total_iou  / processed
        print(f"\n{'=' * 60}")
        print(f"  RISULTATI FINALI ({processed} immagini di test)")
        print(f"{'=' * 60}")
        print(f"  Dice Score medio : {avg_dice:.4f}")
        print(f"  IoU medio        : {avg_iou:.4f}")
        print(f"{'=' * 60}")
    else:
        print("[ATTENZIONE] Nessuna immagine processata.")


if __name__ == "__main__":
    main()
