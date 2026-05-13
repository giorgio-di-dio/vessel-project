import torch
import segmentation_models_pytorch as smp

print(f"Versione PyTorch: {torch.__version__}")
print(f"GPU disponibile: {torch.cuda.is_available()}")

# Prova a inizializzare una U-Net al volo
model = smp.Unet(encoder_name="resnet34", classes=1)
print("Modello U-Net creato con successo!")