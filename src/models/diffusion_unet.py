import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Mappa il timestep scalare in un vettore (embedding) ad alta dimensionalità 
    usando funzioni sinusoidali. Simile agli embedding di posizione in Transformer.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.down = nn.MaxPool2d(2)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        # Prima convoluzione
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # Iniezione del time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2] # Broadcasting spaziale
        h = h + time_emb
        
        # Seconda convoluzione
        h = self.bnorm2(self.relu(self.conv2(h)))
        skip = h # Salviamo la feature map per la skip connection
        
        out = self.down(h)
        return out, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        # L'in_ch qui è il numero di feature map in uscita dal blocco precedente.
        # up campiona queste feature. Poi le concateniamo alla skip connection.
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        
        # Dopo la concatenazione (out_ch + out_ch), facciamo conv
        self.conv1 = nn.Conv2d(out_ch * 2, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, skip, t):
        # Upsampling
        h = self.up(x)
        
        # Concatenate skip connection
        h = torch.cat([h, skip], dim=1)
        
        # Prima convoluzione
        h = self.bnorm1(self.relu(self.conv1(h)))
        
        # Iniezione del time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        
        # Seconda convoluzione
        h = self.bnorm2(self.relu(self.conv2(h)))
        return h

class ConditionalUNet(nn.Module):
    """
    U-Net Leggera condizionata progettata per modelli Diffusion.
    """
    def __init__(self, image_channels=3, mask_channels=1, base_dim=32):
        super().__init__()
        
        # INPUT: L'immagine originale (condizione, 3 canali) e la maschera rumorosa (1 canale)
        # vengono concatenate. In_channels = 3 + 1 = 4.
        in_channels = image_channels + mask_channels
        out_channels = mask_channels # Rete deve predire il rumore della maschera (1 canale)
        
        time_emb_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Encoder (Downsampling) - Architecture: 32 -> 64 -> 128
        self.down1 = DownBlock(in_channels, base_dim, time_emb_dim)
        self.down2 = DownBlock(base_dim, base_dim * 2, time_emb_dim)
        self.down3 = DownBlock(base_dim * 2, base_dim * 4, time_emb_dim)
        
        # Bottleneck - Architecture: 256
        self.bot1 = nn.Conv2d(base_dim * 4, base_dim * 8, 3, padding=1)
        self.bot_time = nn.Linear(time_emb_dim, base_dim * 8)
        self.bot2 = nn.Conv2d(base_dim * 8, base_dim * 8, 3, padding=1)
        
        # Decoder (Upsampling) - Architecture: 128 -> 64 -> 32
        self.up1 = UpBlock(base_dim * 8, base_dim * 4, time_emb_dim)
        self.up2 = UpBlock(base_dim * 4, base_dim * 2, time_emb_dim)
        self.up3 = UpBlock(base_dim * 2, base_dim, time_emb_dim)
        
        # Output Convolution
        self.out = nn.Conv2d(base_dim, out_channels, 1)
        
    def forward(self, x_noisy, x_cond, time):
        """
        Args:
            x_noisy: batch di maschere rumorose (B, 1, H, W)
            x_cond: batch di immagini fondo oculare (B, 3, H, W)
            time: batch di timestep scalari (B,)
        Returns:
            Rumore predetto (B, 1, H, W)
        """
        # --- Conditioning ---
        # Concatena l'immagine originale alla maschera rumorosa
        x = torch.cat([x_noisy, x_cond], dim=1) # Shape: (B, 4, H, W)
        
        # Estrai i Time Embeddings
        t = self.time_mlp(time)
        
        # --- Encoder ---
        x, skip1 = self.down1(x, t)
        x, skip2 = self.down2(x, t)
        x, skip3 = self.down3(x, t)
        
        # --- Bottleneck ---
        x = F.relu(self.bot1(x))
        time_emb = F.relu(self.bot_time(t))
        time_emb = time_emb[(..., ) + (None, ) * 2] # estensione spaziale
        x = x + time_emb
        x = F.relu(self.bot2(x))
        
        # --- Decoder ---
        x = self.up1(x, skip3, t)
        x = self.up2(x, skip2, t)
        x = self.up3(x, skip1, t)
        
        # Output predizione rumore
        return self.out(x)
