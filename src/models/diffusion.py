import torch
import torch.nn as nn
import math

# ==========================================
# CONSTANTS & SCHEDULER
# ==========================================
TIMESTEPS = 300

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(timesteps=TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def p_sample(model, x, cond, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    predicted_noise = model(x, cond, t)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(betas_t) * noise

@torch.no_grad()
def p_sample_loop(model, cond_img):
    device = cond_img.device
    b, c, h, w = cond_img.shape

    # Partiamo da rumore PURO (1 canale)
    img = torch.randn((b, 1, h, w), device=device)

    imgs_intermedie = []

    for i in reversed(range(0, TIMESTEPS)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, cond_img, t, i)

        if i == TIMESTEPS - 1 or i % 50 == 0:
            imgs_intermedie.append(img.cpu())

    # Ritorna l'immagine finale e i passaggi intermedi
    return img, imgs_intermedie

# ==========================================
# ARCHITETTURA (UNet)
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
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

class Block(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        if up:
            self.conv1 = nn.Conv2d(2 * in_c, out_c, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_c, out_c, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
            self.transform = nn.Conv2d(out_c, out_c, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_c)
        self.bnorm2 = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class ConditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3 # Immagine fundus
        mask_channels = 1  # Maschera rumorosa
        in_channels = image_channels + mask_channels # 4 canali in totale
        out_channels = 1 # Rumore predetto

        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([
            Block(down_channels[0], down_channels[1], time_emb_dim),
            Block(down_channels[1], down_channels[2], time_emb_dim),
        ])

        self.ups = nn.ModuleList([
            Block(up_channels[0], up_channels[1], time_emb_dim, up=True),
            Block(up_channels[1], up_channels[2], time_emb_dim, up=True),
        ])

        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

    def forward(self, x, cond, timestep):
        t = self.time_mlp(timestep)
        x = torch.cat([cond, x], dim=1)
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)
