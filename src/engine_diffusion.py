import os
import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch_diffusion(model, dataloader, optimizer, scheduler, scaler, device):
    """
    Esegue un'epoca di addestramento per il modello Diffusion.
    Utilizza Automatic Mixed Precision (AMP) per ridurre l'uso di memoria (fondamentale su Colab).
    """
    model.train()
    running_loss = 0.0
    
    # Usiamo la MSE (Mean Squared Error) perché vogliamo stimare il valore del rumore gaussiano
    criterion = nn.MSELoss()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Assumiamo che il tuo dataloader restituisca tuple (images, masks)
        # Se restituisce dizionari, andrà adattato (es. batch['image'], batch['mask'])
        images = batch[0].to(device)
        masks = batch[1].to(device)
        
        batch_size = images.shape[0]
        
        # 1. Genera timestep casuali indipendenti per ogni immagine nel batch
        t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device).long()
        
        # 2. Processo Forward: corrompi la ground truth (maschera) aggiungendo rumore
        noisy_masks, noise = scheduler.q_sample(masks, t)
        
        optimizer.zero_grad()
        
        # 3. Forward Pass del Modello sotto il context manager autocast (FP16)
        with torch.cuda.amp.autocast(device_type=device):
            # Il modello prova a prevedere il rumore avendo l'immagine condizionale e la maschera rumorosa
            predicted_noise = model(noisy_masks, images, t)
            
            # Calcolo Loss
            loss = criterion(predicted_noise, noise)
            
        # 4. Backward Pass ottimizzato con Scaler (previene problemi di underflow con FP16)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return running_loss / len(dataloader)

def train_diffusion(
    model, 
    dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device, 
    save_dir="checkpoints", 
    drive_save_dir=None
):
    """
    Loop principale per l'addestramento.
    Implementa un sistema di salvataggio ridondante sia locale che, facoltativamente, su Google Drive.
    """
    os.makedirs(save_dir, exist_ok=True)
    if drive_save_dir:
        os.makedirs(drive_save_dir, exist_ok=True)
        
    # Inizializzatore per Mixed Precision
    scaler = torch.cuda.amp.GradScaler(device_type=device)
    best_loss = float('inf')
    
    print(f"Inizio addestramento Diffusion su {device} per {num_epochs} epoche.")
    
    for epoch in range(num_epochs):
        print(f"\nEpoca {epoch+1}/{num_epochs}")
        
        # Lancia l'epoca
        avg_loss = train_epoch_diffusion(model, dataloader, optimizer, scheduler, scaler, device)
        print(f"Loss media dell'epoca: {avg_loss:.6f}")
        
        # Creazione del dizionario di stato per poter fare "resume"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        
        # Salvataggio del checkpoint corrente (sovrascrive quello dell'epoca precedente)
        local_path = os.path.join(save_dir, "diffusion_last.pth")
        torch.save(checkpoint, local_path)
        if drive_save_dir:
            drive_path = os.path.join(drive_save_dir, "diffusion_last.pth")
            torch.save(checkpoint, drive_path)
            
        # Se la loss migliora, salva separatamente come 'best'
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            best_path = os.path.join(save_dir, "diffusion_best.pth")
            torch.save(model.state_dict(), best_path) # Qui salviamo solo i pesi per leggerezza
            
            if drive_save_dir:
                drive_best = os.path.join(drive_save_dir, "diffusion_best.pth")
                torch.save(model.state_dict(), drive_best)
                
            print(f"-> Nuovo modello migliore salvato! (Loss: {best_loss:.6f})")

    print("Addestramento completato.")
