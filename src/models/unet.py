"""
src/models/unet.py
------------------
Implementazione from-scratch dell'architettura U-Net per la segmentazione
semantica di immagini medicali (vasi sanguigni da immagini Fundus).

Architettura originale: Ronneberger et al., 2015
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/abs/1505.04597

Struttura:
    - Encoder (contrattivo): 4 stadi di DoubleConv + MaxPool
    - Bottleneck: DoubleConv al livello più profondo
    - Decoder (espansivo): 4 stadi di Upsample + Concatenazione (skip conn.) + DoubleConv
    - Output Head: convoluzione 1x1 -> 1 canale (logit per sigmoid)

Note implementative:
    - Si usa 'bilinear upsample + conv' invece di 'TransposedConv' per evitare
      il problema dei checkerboard artifacts tipico delle deconvoluzioni.
    - Ogni DoubleConv include BatchNorm per stabilizzare il training.
    - L'output sono logits grezzi (NON applicare sigmoid qui -> già gestita dalla BCEWithLogitsLoss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# BLOCCO ELEMENTARE: DOUBLE CONVOLUTION
# ==============================================================================

class DoubleConv(nn.Module):
    """
    Blocco Conv-BN-ReLU ripetuto due volte: il mattone di base della U-Net.

        Input -> [Conv3x3 -> BatchNorm -> ReLU] x2 -> Output

    Args:
        in_channels  : Numero di canali in ingresso.
        out_channels : Numero di feature maps in uscita.
        mid_channels : Canali intermedi (default = out_channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


# ==============================================================================
# BLOCCHI ENCODER E DECODER
# ==============================================================================

class EncoderBlock(nn.Module):
    """
    Stadio encoder: MaxPool2x2 (dimezza spazialmente) seguito da DoubleConv.
    Corrisponde a un singolo step di downsampling nell'encoder della U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class DecoderBlock(nn.Module):
    """
    Stadio decoder: Upsample bilineare 2x + concatenazione con la skip connection
    dell'encoder + DoubleConv.

    L'upsample bilineare è preferito alla TransposedConv per evitare checkerboard
    artifacts. La skip connection inietta le feature maps dell'encoder corrispondente,
    recuperando le informazioni spaziali ad alta risoluzione perse nel downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # L'upsample raddoppia spazialmente le feature maps
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # in_channels = canali decoder + canali skip connection (già sommati in UNet.__init__)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_decoder : Feature map del decoder (dimensioni dimezzate rispetto a x_encoder).
            x_encoder : Skip connection dall'encoder (dimensioni originali).
        """
        x_decoder = self.up(x_decoder)

        # Gestione delle discrepanze di dimensione dovute a input non perfettamente divisibili
        # (pad il decoder per matchare la dimensione dell'encoder)
        diff_H = x_encoder.shape[2] - x_decoder.shape[2]
        diff_W = x_encoder.shape[3] - x_decoder.shape[3]
        x_decoder = F.pad(
            x_decoder,
            [diff_W // 2, diff_W - diff_W // 2, diff_H // 2, diff_H - diff_H // 2],
        )

        # Concatenazione lungo la dimensione dei canali: (B, C_dec+C_enc, H, W)
        x = torch.cat([x_encoder, x_decoder], dim=1)
        return self.conv(x)


# ==============================================================================
# ARCHITETTURA U-NET COMPLETA
# ==============================================================================

class UNet(nn.Module):
    """
    U-Net per segmentazione binaria (es. vasi sanguigni vs background).

    Encoder:
        in_channels -> 64 -> 128 -> 256 -> 512 (+ MaxPool 2x tra ogni stadio)

    Bottleneck:
        512 -> 1024

    Decoder (con skip connections dall'encoder):
        1024+512 -> 512 -> 256 -> 128 -> 64 (+ Upsample 2x tra ogni stadio)

    Output Head:
        64 -> out_channels (1 per segmentazione binaria)
        NOTA: Restituisce logits grezzi, NON probabilità (nessuna sigmoid).
              La sigmoid è applicata dalla loss (BCEWithLogitsLoss) o durante
              l'inference in analyze.py.

    Args:
        in_channels  : Canali di input (3 per RGB).
        out_channels : Canali di output (1 per maschera binaria).
        features     : Lista di feature map per ogni stadio dell'encoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list = [64, 128, 256, 512],
    ):
        super().__init__()

        # --- ENCODER ---
        self.input_conv = DoubleConv(in_channels, features[0])
        self.enc1 = EncoderBlock(features[0], features[1]) # 64 -> 128
        self.enc2 = EncoderBlock(features[1], features[2]) # 128 -> 256
        self.enc3 = EncoderBlock(features[2], features[3]) # 256 -> 512

        # --- BOTTLENECK ---
        self.bottleneck = EncoderBlock(features[3], features[3] * 2) # 512 -> 1024

        # --- DECODER ---
        # Dopo l'upsample, i canali del decoder vengono concatenati con i canali
        # della skip connection corrispondente. Il canale totale in ingresso al
        # DoubleConv successivo è quindi: canali_decoder + canali_skip.
        self.dec3 = DecoderBlock(features[3] * 2 + features[3], features[3])  # 1024+512=1536 -> 512
        self.dec2 = DecoderBlock(features[3] + features[2],      features[2])  # 512+256=768   -> 256
        self.dec1 = DecoderBlock(features[2] + features[1],      features[1])  # 256+128=384   -> 128
        self.dec0 = DecoderBlock(features[1] + features[0],      features[0])  # 128+64=192    -> 64

        # --- OUTPUT HEAD ---
        # Convoluzione 1x1: riduce da features[0] canali a out_channels logits
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass della U-Net.

        Args:
            x : Input tensor di forma (B, in_channels, H, W).

        Returns:
            Logits di forma (B, out_channels, H, W). Applicare torch.sigmoid()
            per ottenere probabilità durante l'inference.
        """
        # Encoder (salviamo ogni skip connection)
        s0 = self.input_conv(x)     # (B, 64, H, W)
        s1 = self.enc1(s0)          # (B, 128, H/2, W/2)
        s2 = self.enc2(s1)          # (B, 256, H/4, W/4)
        s3 = self.enc3(s2)          # (B, 512, H/8, W/8)

        # Bottleneck
        b  = self.bottleneck(s3)    # (B, 1024, H/16, W/16)

        # Decoder (con skip connections)
        x = self.dec3(b, s3)        # (B, 512, H/8, W/8)
        x = self.dec2(x, s2)        # (B, 256, H/4, W/4)
        x = self.dec1(x, s1)        # (B, 128, H/2, W/2)
        x = self.dec0(x, s0)        # (B, 64, H, W)

        # Output head
        return self.output_conv(x)  # (B, 1, H, W) logits

    def count_parameters(self) -> int:
        """Restituisce il numero totale di parametri trainabili."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# TEST RAPIDO DELL'ARCHITETTURA
# ==============================================================================

if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    print(f"[UNet] Parametri trainabili: {model.count_parameters():,}")

    # Simula un batch di 2 patch 512x512
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"[UNet] Input shape:  {tuple(dummy_input.shape)}")
    print(f"[UNet] Output shape: {tuple(output.shape)}")
    assert output.shape == (2, 1, 512, 512), "Errore: shape dell'output non corretto!"
    print("[UNet] Test shape: OK")
