"""
src/metrics.py
--------------
Metriche di valutazione per la segmentazione binaria dei vasi sanguigni.

Metriche implementate:
    - Dice Score (F1 per segmentazione): 2*TP / (2*TP + FP + FN)
    - IoU / Jaccard Index: TP / (TP + FP + FN)
    - Pixel Accuracy: (TP + TN) / (TP + TN + FP + FN)

Nota: Tutte le metriche operano su tensori binarizzati con soglia MASK_THRESHOLD.
      I valori di input possono essere logits grezzi o probabilità (specificare from_logits).
"""

import math
import torch
import torch.nn.functional as F
from src.config import MASK_THRESHOLD


# ==============================================================================
# METRICHE DI BASE
# ==============================================================================

def dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = MASK_THRESHOLD,
    smooth: float = 1.0,
    from_logits: bool = True,
) -> float:
    """
    Calcola il Dice Score (coefficiente F1 per segmentazione) per un batch.

    Formula: Dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)

    Args:
        predictions : Logits o probabilità (B, 1, H, W).
        targets     : Ground truth binarie (B, 1, H, W), valori in {0.0, 1.0}.
        threshold   : Soglia per binarizzare le probabilità predette.
        smooth      : Smoothing per evitare divisione per zero.
        from_logits : Se True, applica sigmoid prima della sogliatura.

    Returns:
        Dice Score medio sul batch (float in [0, 1]).
    """
    with torch.no_grad():
        if from_logits:
            predictions = torch.sigmoid(predictions)

        # Binarizzazione: P_bin in {0, 1}
        pred_binary = (predictions > threshold).float()
        targets_flat = targets.view(targets.shape[0], -1).float()
        pred_flat    = pred_binary.view(pred_binary.shape[0], -1)

        intersection = (pred_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (
            pred_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth
        )
        return dice.mean().item()


def iou_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = MASK_THRESHOLD,
    smooth: float = 1.0,
    from_logits: bool = True,
) -> float:
    """
    Calcola l'IoU (Intersection over Union), anche detto Jaccard Index.

    Formula: IoU = (TP + smooth) / (TP + FP + FN + smooth)

    L'IoU è più conservativo del Dice Score perché non conta il doppio dei TP
    al numeratore. Un IoU di 0.7 corrisponde approssimativamente a un Dice di ~0.82.

    Args:
        predictions : Logits o probabilità (B, 1, H, W).
        targets     : Ground truth binarie (B, 1, H, W).
        threshold   : Soglia per binarizzare le probabilità.
        smooth      : Smoothing anti-zero.
        from_logits : Se True, applica sigmoid prima della sogliatura.

    Returns:
        IoU medio sul batch (float in [0, 1]).
    """
    with torch.no_grad():
        if from_logits:
            predictions = torch.sigmoid(predictions)

        pred_binary  = (predictions > threshold).float()
        targets_flat = targets.view(targets.shape[0], -1).float()
        pred_flat    = pred_binary.view(pred_binary.shape[0], -1)

        intersection = (pred_flat * targets_flat).sum(dim=1)
        union        = pred_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
        iou          = (intersection + smooth) / (union + smooth)
        return iou.mean().item()


def pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = MASK_THRESHOLD,
    from_logits: bool = True,
) -> float:
    """
    Calcola la Pixel Accuracy: percentuale di pixel classificati correttamente.

    Formula: Acc = (TP + TN) / (TP + TN + FP + FN)

    ATTENZIONE: In dataset sbilanciati, questa metrica è fuorviante (la rete
    può ottenere 95%+ di accuratezza predendo tutto background). Usarla sempre
    in affiancamento a Dice Score e IoU.

    Returns:
        Pixel accuracy media sul batch (float in [0, 1]).
    """
    with torch.no_grad():
        if from_logits:
            predictions = torch.sigmoid(predictions)

        pred_binary = (predictions > threshold).float()
        correct     = (pred_binary == targets).float()
        return correct.mean().item()


# ==============================================================================
# DISTANZE E METRICHE DI BORDO
# ==============================================================================

def get_boundary(mask: torch.Tensor) -> torch.Tensor:
    """
    Estrae il bordo (contorno) di una maschera binaria.
    Args:
        mask: Tensor di forma (B, 1, H, W) con valori {0, 1}
    Returns:
        Tensor di forma (B, 1, H, W) con valori {0, 1} rappresentanti il bordo.
    """
    eroded = 1 - F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
    return mask - eroded

import numpy as np
from scipy.ndimage import distance_transform_edt

def average_hausdorff_distance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = MASK_THRESHOLD,
    from_logits: bool = True,
) -> float:
    """
    Calcola l'Average Hausdorff Distance (AHD) tra i bordi predetti e quelli reali.
    
    Usa la Trasformata di Distanza (EDT) per evitare di allocare matrici
    di distanze NxM che saturerebbero la memoria RAM (OOM) su immagini 2048x2048.
    
    Formula: max( mean(min_dist(A, B)), mean(min_dist(B, A)) )
    
    Args:
        predictions : Logits o probabilità (B, 1, H, W).
        targets     : Ground truth binarie (B, 1, H, W).
        threshold   : Soglia per binarizzare le probabilità.
        from_logits : Se True, applica sigmoid prima della sogliatura.
        
    Returns:
        Average Hausdorff Distance media sul batch in pixel. Più è bassa, meglio è.
    """
    with torch.no_grad():
        if from_logits:
            predictions = torch.sigmoid(predictions)
            
        pred_bin = (predictions > threshold).float()
        tgt_bin = targets.float()
        
        bound_pred = get_boundary(pred_bin)
        bound_tgt = get_boundary(tgt_bin)
        
        B, _, H, W = pred_bin.shape
        max_dist = math.sqrt(H**2 + W**2)  # Distanza massima possibile (diagonale)
        
        bound_pred_np = bound_pred.cpu().numpy()
        bound_tgt_np = bound_tgt.cpu().numpy()
        
        ahd_batch = []
        
        for i in range(B):
            bp = bound_pred_np[i, 0]
            bt = bound_tgt_np[i, 0]
            
            if bp.sum() == 0 and bt.sum() == 0:
                ahd_batch.append(0.0)
            elif bp.sum() == 0 or bt.sum() == 0:
                ahd_batch.append(max_dist)
            else:
                # edt accetta un array con 0 sui target e 1 altrove.
                # bp e bt hanno 1 sul bordo e 0 altrove, quindi facciamo 1 - array
                edt_p = distance_transform_edt(1 - bp)
                edt_t = distance_transform_edt(1 - bt)
                
                # Distanza media dai punti predetti ai ground truth
                d_p_t = edt_t[bp == 1].mean()
                # Distanza media dai punti ground truth ai predetti
                d_t_p = edt_p[bt == 1].mean()
                
                ahd_batch.append(max(d_p_t, d_t_p))
                
        return float(np.mean(ahd_batch))

