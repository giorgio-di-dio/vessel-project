import os
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from src.config import (
    DEVICE,
    MASK_THRESHOLD,
    PATCH_SIZE,
    create_directories,
    get_kaggle_dataset_path,
)
from src.models import UNet, ConditionalUNet, p_sample_loop
from src.metrics import dice_score, iou_score, average_hausdorff_distance
from analyze import sliding_window_inference

# Modelli paths
UNET_MODEL_PATH = Path("output/compare/best_unet_vessel.pth")
DIFFUSION_MODEL_PATH = Path("output/compare/diffusion_checkpoint_epoch_110.pth")

# Transformazioni per il modello di diffusione come in training
img_size = 256
transform_img = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_mask = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

def denormalize_diffusion_output(tensor):
    img = (tensor + 1) / 2
    img = img.clamp(0, 1)
    return img

def main():
    create_directories()
    print("=" * 60)
    print("  VESSEL SEGMENTATION - COMPARISON: CNN vs DIFFUSION")
    print("=" * 60)

    # Verifica checkpoint
    if not UNET_MODEL_PATH.exists():
        print(f"[ERRORE] File dei pesi mancanti: {UNET_MODEL_PATH}")
        return
    if not DIFFUSION_MODEL_PATH.exists():
        print(f"[ERRORE] File dei pesi mancanti: {DIFFUSION_MODEL_PATH}")
        return

    # Inizializza e carica U-Net
    print("Caricamento CNN...")
    unet_model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    unet_checkpoint = torch.load(UNET_MODEL_PATH, map_location=DEVICE, weights_only=True)
    unet_model.load_state_dict(unet_checkpoint["state_dict"])
    unet_model.eval()

    # Inizializza e carica Diffusion Model
    print("Caricamento Diffusion Model...")
    diffusion_model = ConditionalUNet().to(DEVICE)
    diffusion_checkpoint = torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE, weights_only=True)
    diffusion_model.load_state_dict(diffusion_checkpoint["model_state_dict"])
    diffusion_model.eval()

    # Percorsi test (prendiamo una singola immagine di test)
    dataset_root = get_kaggle_dataset_path()
    test_img_dir = dataset_root / "test" / "Original"
    test_mask_dir = dataset_root / "test" / "Ground truth"

    image_paths = sorted(test_img_dir.glob("*.png"))
    if not image_paths:
        print("[ERRORE] Nessuna immagine di test trovata.")
        return

    img_path = image_paths[0]
    mask_path = test_mask_dir / img_path.name
    
    print(f"\nUso immagine: {img_path.name}")

    # Caricamento immagine
    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Adatta l'immagine e maschera per Diffusion (resize 256x256)
    image_rgb_resized = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    gt_mask_resized = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

    # --- INFERENZA CNN (Sliding Window su Immagine Originale) ---
    print("Avvio inferenza CNN (Sliding Window)...")
    pred_map_unet = sliding_window_inference(
        unet_model, image_rgb, patch_size=PATCH_SIZE, stride=384, device=DEVICE
    )

    # --- INFERENZA DIFFUSION (su Immagine Resized) ---
    pil_img = Image.fromarray(image_rgb_resized)
    img_tensor_diff = transform_img(pil_img).unsqueeze(0).to(DEVICE)
    
    print("Avvio inferenza modello di diffusione (300 step)...")
    with torch.no_grad():
        diff_pred_tensor, _ = p_sample_loop(diffusion_model, img_tensor_diff)
        diff_pred_tensor = denormalize_diffusion_output(diff_pred_tensor)
        pred_map_diff = diff_pred_tensor.squeeze().cpu().numpy()

    # --- CALCOLO METRICHE ---
    gt_tensor_unet = torch.from_numpy((gt_mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    pred_tensor_unet = torch.from_numpy(pred_map_unet).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # Metriche CNN (sull'originale 2048x2048)
    unet_dice = dice_score(pred_tensor_unet, gt_tensor_unet, from_logits=False)
    unet_iou = iou_score(pred_tensor_unet, gt_tensor_unet, from_logits=False)
    unet_ahd = average_hausdorff_distance(pred_tensor_unet, gt_tensor_unet, from_logits=False)

    # Metriche Diffusion (sul resized 256x256)
    gt_tensor_diff = torch.from_numpy((gt_mask_resized > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    diff_dice = dice_score(diff_pred_tensor, gt_tensor_diff, threshold=0.5, from_logits=False)
    diff_iou = iou_score(diff_pred_tensor, gt_tensor_diff, threshold=0.5, from_logits=False)
    diff_ahd = average_hausdorff_distance(diff_pred_tensor, gt_tensor_diff, threshold=0.5, from_logits=False)

    print(f"\n{'=' * 40}")
    print(f" RISULTATI METRICHE ({img_path.name})")
    print(f"{'=' * 40}")
    print(f"CNN (Sliding Window 2048x2048):")
    print(f"  Dice Score        : {unet_dice:.4f}")
    print(f"  IoU               : {unet_iou:.4f}")
    print(f"  Average Hausdorff : {unet_ahd:.4f} pixel")
    print(f"\nDiffusion Model:")
    print(f"  Dice Score        : {diff_dice:.4f}")
    print(f"  IoU               : {diff_iou:.4f}")
    print(f"  Average Hausdorff : {diff_ahd:.4f} pixel")
    print(f"{'=' * 40}")

    # --- SALVATAGGIO GRAFICO COMPARATIVO ---
    pred_binary_unet = (pred_map_unet > MASK_THRESHOLD).astype(np.uint8) * 255
    pred_binary_diff = (pred_map_diff > 0.5).astype(np.uint8) * 255
    gt_binary_unet = (gt_mask > 127).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor("#1a1a2e")

    titles = ["Immagine Originale", "Ground Truth", "Predizione CNN", "Predizione Diffusion"]
    images = [image_rgb, gt_binary_unet, pred_binary_unet, pred_binary_diff]
    cmaps = [None, "gray", "gray", "gray"]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=15, pad=10)
        ax.axis("off")

    output_dir = Path("output/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"compare_result_{img_path.stem}.png"
    
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=80, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[Grafico] Risultato comparativo salvato in {output_path}")

if __name__ == "__main__":
    main()
