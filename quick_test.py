"""
quick_test.py
-------------
Verifica rapida visiva del modello addestrato su N immagini casuali dal test set.

Salva le immagini comparative in output/results/quick_test_<nome>.png
e stampa Dice Score e IoU per ogni immagine.

Esecuzione:
    python quick_test.py          -> testa 3 immagini (default)
    python quick_test.py --n 5    -> testa 5 immagini
"""

import argparse
import random
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    DEVICE,
    PATCH_SIZE,
    MASK_THRESHOLD,
    BEST_MODEL_PATH,
    RESULTS_DIR,
    RANDOM_SEED,
    create_directories,
    get_kaggle_dataset_path,
)
from src.models.unet import UNet
from src.engine import load_checkpoint
from src.metrics import dice_score, iou_score


# ==============================================================================
# SLIDING WINDOW INFERENCE (locale, stride 384)
# ==============================================================================

def sliding_window_inference(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_SIZE // 2,   # stride ≤ patch_size per coprire tutta l'immagine
    device: str = DEVICE,
) -> np.ndarray:
    H, W = image_rgb.shape[:2]
    model.eval()

    prediction_map = np.zeros((H, W), dtype=np.float32)
    count_map      = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for r in range(0, H - patch_size + 1, stride):
            for c in range(0, W - patch_size + 1, stride):
                patch = image_rgb[r:r + patch_size, c:c + patch_size]
                t = (patch.astype(np.float32) / 255.0)
                t = np.transpose(t, (2, 0, 1))
                t = torch.from_numpy(t).unsqueeze(0).to(device)

                logit = model(t)
                prob  = torch.sigmoid(logit).squeeze().cpu().numpy()

                prediction_map[r:r + patch_size, c:c + patch_size] += prob
                count_map[r:r + patch_size, c:c + patch_size]       += 1.0

    count_map = np.maximum(count_map, 1.0)
    return prediction_map / count_map


# ==============================================================================
# VISUALIZZAZIONE
# ==============================================================================

def save_comparison(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_map: np.ndarray,
    output_path: Path,
    title: str,
    dice: float,
    iou: float,
) -> None:
    pred_binary = (pred_map > MASK_THRESHOLD).astype(np.uint8) * 255
    gt_binary   = (gt_mask > 127).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"{title}  |  Dice: {dice:.4f}  |  IoU: {iou:.4f}",
        color="white", fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, img, cmap, label in zip(
        axes,
        [image_rgb, gt_binary, pred_binary],
        [None, "gray", "gray"],
        ["Immagine Originale", "Ground Truth", "Predizione U-Net"],
    ):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label, color="white", fontsize=12, pad=8)
        ax.axis("off")

    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=80, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Salvato: {output_path.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main(n: int, stride: int = PATCH_SIZE // 2):
    create_directories()

    # --- Verifica checkpoint ---
    if not BEST_MODEL_PATH.exists():
        print(f"[ERRORE] Checkpoint non trovato: {BEST_MODEL_PATH}")
        print("         Lancia prima: python main.py")
        return

    # --- Caricamento modello ---
    # Leggiamo prima i features dall'checkpoint (se disponibili) per ricostruire
    # l'architettura esatta usata durante il training ed evitare size mismatch.
    _ckpt_meta = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=False)
    _features  = _ckpt_meta.get("features", None)

    if _features is None:
        # Checkpoint vecchio: deduciamo i features dallo state_dict
        _sd = _ckpt_meta["state_dict"]
        _f0 = _sd["input_conv.double_conv.0.weight"].shape[0]   # enc0 out
        _f1 = _sd["enc1.maxpool_conv.1.double_conv.0.weight"].shape[0]
        _f2 = _sd["enc2.maxpool_conv.1.double_conv.0.weight"].shape[0]
        _f3 = _sd["enc3.maxpool_conv.1.double_conv.0.weight"].shape[0]
        _features = [_f0, _f1, _f2, _f3]
        print(f"  [Info] Features dedotti dal checkpoint: {_features}")
    else:
        print(f"  [Info] Features caricati dal checkpoint: {_features}")

    model = UNet(in_channels=3, out_channels=1, features=_features).to(DEVICE)
    load_checkpoint(model, BEST_MODEL_PATH, device=DEVICE)
    model.eval()
    print(f"\nDevice: {DEVICE.upper()} | Patch size: {PATCH_SIZE} | Stride: {args.stride}\n")

    # --- Percorsi test set ---
    dataset_root  = get_kaggle_dataset_path()
    test_img_dir  = dataset_root / "test" / "Original"
    test_mask_dir = dataset_root / "test" / "Ground truth"

    all_images = sorted(test_img_dir.glob("*.png"))
    # Filtra solo le immagini con maschera disponibile
    valid = [(p, test_mask_dir / p.name) for p in all_images if (test_mask_dir / p.name).exists()]

    if not valid:
        print(f"[ERRORE] Nessuna coppia immagine/maschera trovata in {test_img_dir}")
        return

    # Selezione casuale di N immagini
    random.seed(RANDOM_SEED)
    selected = random.sample(valid, min(n, len(valid)))
    print(f"Testando {len(selected)} immagini casuali dal test set ({len(valid)} disponibili):\n")

    results = []

    for img_path, mask_path in selected:
        print(f"  [{img_path.name}]")

        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_mask   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Inferenza sliding window
        pred_map = sliding_window_inference(model, image_rgb, PATCH_SIZE, stride=args.stride, device=DEVICE)

        # Metriche
        pred_t = torch.from_numpy(pred_map).unsqueeze(0).unsqueeze(0)
        gt_t   = torch.from_numpy((gt_mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        d = dice_score(pred_t, gt_t, from_logits=False)
        i = iou_score(pred_t, gt_t, from_logits=False)
        print(f"     Dice: {d:.4f} | IoU: {i:.4f}")
        results.append((d, i))

        # Salva figura
        out_path = RESULTS_DIR / f"quick_test_{img_path.stem}.png"
        save_comparison(image_rgb, gt_mask, pred_map, out_path, img_path.stem, d, i)

    # --- Riepilogo ---
    avg_dice = sum(r[0] for r in results) / len(results)
    avg_iou  = sum(r[1] for r in results) / len(results)
    print(f"\n{'=' * 40}")
    print(f"  Media su {len(results)} immagini")
    print(f"  Dice Score : {avg_dice:.4f}")
    print(f"  IoU        : {avg_iou:.4f}")
    print(f"{'=' * 40}")
    print(f"\nImmagini salvate in: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test visivo del modello U-Net.")
    parser.add_argument("--n",      type=int, default=3,              help="Numero di immagini casuali da testare (default: 1)")
    parser.add_argument("--stride", type=int, default=PATCH_SIZE // 2,
                        help=f"Stride per sliding window (default: {PATCH_SIZE // 2} = patch_size/2, overlap 50%%). "
                             f"Deve essere ≤ patch_size ({PATCH_SIZE}) per evitare gap non coperti.")
    args = parser.parse_args()

    if args.stride > PATCH_SIZE:
        print(f"[ATTENZIONE] stride ({args.stride}) > patch_size ({PATCH_SIZE}): "
              f"ci saranno pixel non coperti → strisce nere a griglia!")

    main(args.n, args.stride)
