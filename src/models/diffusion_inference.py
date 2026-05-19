import torch
import torch.nn.functional as F
from tqdm import tqdm

class DiffusionPipeline:
    """
    Gestisce l'intero processo di Inferenza: Reverse Sampling (Denoising) e 
    ricostruzione (stitching) dell'immagine finale ad alta risoluzione.
    """
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.model.eval() # Assicuriamoci che il modello sia in modalità valutazione
        
    @torch.no_grad()
    def p_sample(self, x, x_cond, t, t_index):
        """
        Singolo step del Reverse Process (Standard DDPM).
        """
        # Estraiamo i coefficienti per il timestep corrente
        betas_t = self.scheduler._extract(self.scheduler.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.scheduler._extract(
            self.scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.scheduler._extract(
            1.0 / torch.sqrt(self.scheduler.alphas), t, x.shape
        )
        
        # 1. La U-Net predice il rumore (condizionata dall'immagine originale x_cond)
        predicted_noise = self.model(x, x_cond, t)
        
        # 2. Rimuoviamo parte del rumore per ottenere l'immagine al timestep t-1
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        # Se siamo all'ultimo step (t=0), restituiamo semplicemente la media
        if t_index == 0:
            return model_mean
        else:
            # Altrimenti aggiungiamo un po' di rumore per stabilità (Langevin Dynamics)
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(betas_t)
            return model_mean + sigma_t * noise
    @torch.no_grad()
    def ddim_sample(self, x, x_cond, t, t_prev):
        """
        Singolo step del Reverse Process usando DDIM (Deterministico).
        Questa variante permette di saltare molti step intermedi accelerando l'inferenza.
        """
        b = x.shape[0]
        # Estraiamo le alpha_cumprod per il timestep t e per il precedente t_prev
        alpha_t = self.scheduler._extract(self.scheduler.alphas_cumprod, t, x.shape)
        
        # Gestiamo il caso dell'ultimo step in cui t_prev scende sotto 0 (immaginiamo alpha_prev = 1)
        if t_prev[0].item() < 0:
            alpha_prev = torch.ones_like(alpha_t)
        else:
            alpha_prev = self.scheduler._extract(self.scheduler.alphas_cumprod, t_prev, x.shape)
            
        # 1. La U-Net predice il rumore
        predicted_noise = self.model(x, x_cond, t)
        
        # 2. Stimiamo l'immagine pulita x_0 sottraendo il rumore
        pred_x0 = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        # 3. Calcoliamo la direzione verso l'immagine al tempo t_prev
        dir_xt = torch.sqrt(1.0 - alpha_prev) * predicted_noise
        
        # 4. Calcoliamo x al tempo t_prev senza aggiungere rumore aggiuntivo (sigma=0 in DDIM)
        x_prev = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
        
        return x_prev

    @torch.no_grad()
    def generate_mask(self, x_cond, use_ddim=False, ddim_steps=50):
        """
        Processo iterativo di generazione della patch (da rumore puro a maschera pulita).
        """
        b, c, h, w = x_cond.shape
        # Si parte da puro rumore gaussiano
        x = torch.randn((b, 1, h, w), device=self.device)
        
        # Controllo della flag per il fast sampler (DDIM)
        if use_ddim:
            # Creiamo la sequenza di step per DDIM (saltando gli intermedi)
            step_ratio = self.scheduler.num_timesteps // ddim_steps
            timesteps = torch.arange(0, ddim_steps) * step_ratio
            timesteps = torch.flip(timesteps, dims=[0]) # Da t_max a 0
            
            for i, step in enumerate(tqdm(timesteps, desc=f"DDIM Sampling ({ddim_steps} steps)", leave=False)):
                step_val = step.item()
                t = torch.full((b,), step_val, device=self.device, dtype=torch.long)
                
                # Calcoliamo il timestep precedente
                prev_step_val = step_val - step_ratio if i < len(timesteps) - 1 else -1
                t_prev = torch.full((b,), prev_step_val, device=self.device, dtype=torch.long)
                
                x = self.ddim_sample(x, x_cond, t, t_prev)
        else:
            # Loop Standard DDPM (tipicamente 1000 step)
            for i in tqdm(reversed(range(0, self.scheduler.num_timesteps)), desc="DDPM Sampling", total=self.scheduler.num_timesteps, leave=False):
                t = torch.full((b,), i, device=self.device, dtype=torch.long)
                x = self.p_sample(x, x_cond, t, i)
            
        return x

    @torch.no_grad()
    def infer_full_image(self, full_image_tensor, patch_size=128, stride=64, use_ddim=False):
        """
        Logica di Stitching/Sliding Window per processare immagini ad alta risoluzione.
        Le patch generate vengono mediate dove si sovrappongono.
        
        Args:
            full_image_tensor: tensore (1, 3, H, W)
            patch_size: Dimensione della patch usata in training
            stride: Se minore di patch_size crea sovrapposizione e migliora i bordi (media).
        """
        assert full_image_tensor.ndim == 4 and full_image_tensor.size(0) == 1, "Invia un'immagine per volta (1, 3, H, W)"
        _, _, H, W = full_image_tensor.shape
        
        # Tensore vuoto per accumulare i risultati
        output_mask = torch.zeros((1, 1, H, W), device=self.device)
        count_map = torch.zeros((1, 1, H, W), device=self.device)
        
        # Estrai i punti iniziali Y e X per le patch
        y_steps = list(range(0, H - patch_size + 1, stride))
        x_steps = list(range(0, W - patch_size + 1, stride))
        
        # Aggiungiamo i bordi finali per assicurarci di coprire tutto
        if y_steps[-1] != H - patch_size: y_steps.append(H - patch_size)
        if x_steps[-1] != W - patch_size: x_steps.append(W - patch_size)
        
        total_patches = len(y_steps) * len(x_steps)
        print(f"Esecuzione inferenza su {total_patches} patches...")
        
        # Ciclo sulla Sliding Window
        for y in tqdm(y_steps, desc="Righe (Y)"):
            for x in x_steps:
                # Estrai patch dell'immagine originale
                patch_cond = full_image_tensor[:, :, y:y+patch_size, x:x+patch_size]
                
                # Genera maschera per la patch
                patch_mask = self.generate_mask(patch_cond, use_ddim=use_ddim)
                
                # Aggiungi al tensore globale (verrà poi diviso per il count_map)
                output_mask[:, :, y:y+patch_size, x:x+patch_size] += patch_mask
                count_map[:, :, y:y+patch_size, x:x+patch_size] += 1
                
        # Media i risultati nelle zone di sovrapposizione
        output_mask = output_mask / torch.clamp(count_map, min=1.0)
        
        # Poiché il modello prevede il rumore latente che potrebbe non essere vincolato,
        # passiamo l'output finale attraverso una Sigmoide per avere la probabilità di classe (0-1).
        final_probability_mask = torch.sigmoid(output_mask)
        return final_probability_mask
