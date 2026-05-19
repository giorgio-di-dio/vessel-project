import torch
import math

class DiffusionScheduler:
    """
    Gestisce la programmazione del rumore (noise schedule) e il processo di forward
    per un modello di diffusione standard (DDPM).
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Schedule lineare per le beta
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Pre-calcoliamo i coefficienti per il Forward Process q(x_t | x_0)
        # x_t = sqrt(alphas_cumprod) * x_0 + sqrt(1 - alphas_cumprod) * noise
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def _extract(self, a, t, x_shape):
        """
        Estrae i coefficienti appropriati per il batch di timestep correnti
        e ne fa il reshape in modo che possano essere moltiplicati con il tensore dell'immagine.
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(-1, t)
        # Reshape: [batch_size, 1, 1, 1] per permettere il broadcasting
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward Process: corrompe x_start (immagine/maschera originale) aggiungendo rumore.
        
        Args:
            x_start: Il tensore originale (es. maschera ground truth).
            t: Tensore di timestep casuali per ogni elemento della batch.
            noise: Il rumore gaussiano (se None, viene generato).
            
        Returns:
            x_t: L'immagine corrotta al timestep t.
            noise: Il rumore che è stato aggiunto (da usare come target per la loss della U-Net).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
