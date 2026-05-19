"""
src/losses.py
-------------
Funzioni di Loss per la segmentazione semantica binaria.

Problema del class imbalance:
    Nelle immagini Fundus, i pixel dei vasi sanguigni rappresentano una
    frazione molto ridotta dell'immagine (~5-15%). Una BCE standard
    convergerebbe a predire "tutto background", ottenendo un'accuratezza
    alta ma una segmentazione nulla.

Soluzione:
    Combiniamo BCE con Dice Loss (CombinedLoss) per forzare la rete a
    focalizzarsi sulla sovrapposizione spaziale della maschera predetta
    con quella ground truth, indipendentemente dallo sbilanciamento.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# DICE LOSS
# ==============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss per segmentazione binaria.

    Il Dice Coefficient misura la sovrapposizione tra maschera predetta e ground truth:
        Dice = (2 * |P ∩ G|) / (|P| + |G|)
    dove P = predizione, G = ground truth.

    Il valore è in [0, 1] con 1 = sovrapposizione perfetta.
    La Dice Loss è definita come: L_dice = 1 - Dice

    Gestione del caso degenere:
        Se sia P che G sono tutti zero (nessun vaso nell'immagine), il coefficiente
        sarebbe 0/0. Il termine 'smooth' al numeratore e denominatore evita la
        divisione per zero e stabilizza il gradiente.

    Args:
        smooth  : Termine di smoothing per evitare divisione per zero (Laplace smoothing). 
        Corregge il denominatore: Dice = (2 * Intersezione + smooth) / (Somma predetti + Somma reali + smooth). 
        In questo modo quando sia GT che la predizione sono zero, la loss vale 1. 
        from_logits : Se True, applica sigmoid all'input prima del calcolo.
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions : Logits o probabilità di forma (B, 1, H, W).
            targets     : Maschere binarie ground truth di forma (B, 1, H, W), valori in {0.0, 1.0}.

        Returns:
            Scalare: valore della Dice Loss mediato sul batch.
        """
        if self.from_logits:
            predictions = torch.sigmoid(predictions)

        # Appiattimento spaziale per il calcolo vettoriale: (B, H*W)
        predictions = predictions.view(predictions.shape[0], -1)
        targets     = targets.view(targets.shape[0], -1)

        # Calcolo dell'intersezione e dell'unione
        intersection = (predictions * targets).sum(dim=1)                 # (B,)
        dice_coeff   = (2.0 * intersection + self.smooth) / (
            predictions.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )                                                                   # (B,)

        # Loss = 1 - Dice (media sul batch)
        return 1.0 - dice_coeff.mean()


# ==============================================================================
# COMBINED LOSS: BCE + DICE
# ==============================================================================

class CombinedLoss(nn.Module):
    """
    Loss combinata: alpha * BCE + (1 - alpha) * Dice.

    La BCE penalizza ogni singolo pixel errato.
    La Dice Loss penalizza globalmente la scarsa sovrapposizione delle maschere.
    Combinarle fornisce sia localizzazione pixel-level che coerenza globale.

    Args:
        alpha       : Peso della BCE nella loss combinata (default=0.5).
        smooth      : Smoothing per la DiceLoss.
        from_logits : Input sono logits grezzi (default=True, usa BCEWithLogitsLoss).
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth, from_logits=from_logits)
        # BCEWithLogitsLoss = Sigmoid + BCE fused, numericamente più stabile
        self.bce_loss  = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions : Logits grezzi (B, 1, H, W).
            targets     : Ground truth binarie (B, 1, H, W), valori in {0.0, 1.0}.

        Returns:
            Scalare: loss combinata pesata.
        """
        bce  = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.alpha * bce + (1.0 - self.alpha) * dice
