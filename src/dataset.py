"""
src/dataset.py
--------------
Gestione del dataset di segmentazione dei vasi sanguigni (Fundus Images).

Struttura dati attesa da Kaggle:
    <root>/
        train/
            Original/       -> immagini RGB (2048x2048, .png)
            Ground truth/   -> maschere binarie (2048x2048, .png)
        test/
            Original/
            Ground truth/

Strategia: Patch-based Random Sampling
    - Durante il training, ogni __getitem__ estrae una patch 512x512 casuale
      da un'immagine casuale, garantendo massima varietà e compatibilità VRAM.
    - Patch size = 512 -> multiplo di 2^5=32, compatibile con U-Net a 5 livelli.
    - Durante l'inference, si usa la funzione sliding_window_inference() in analyze.py.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    PATCH_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
)


# ==============================================================================
# CLASSE PRINCIPALE DEL DATASET
# ==============================================================================

class VesselDataset(Dataset):
    """
    Dataset PyTorch per la segmentazione dei vasi sanguigni su immagini Fundus.

    Args:
        image_dir   : Path alla cartella 'Original' contenente le immagini RGB.
        mask_dir    : Path alla cartella 'Ground truth' contenente le maschere.
        patch_size  : Lato (in pixel) della patch quadrata estratta per il training.
                      Deve essere un multiplo di 32 per garantire compatibilità
                      con U-Net a 5 livelli di profondità (downsampling 2^5).
        mode        : 'train' -> estrae patch random | 'test' -> restituisce l'immagine intera.
        augment     : Se True, applica augmentation geometrica random durante il training.
        extensions  : Estensioni file immagine accettate.
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        patch_size: int = PATCH_SIZE,
        mode: str = "train",
        augment: bool = True,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ):
        assert mode in ("train", "test"), "Il parametro 'mode' deve essere 'train' o 'test'."
        assert patch_size % 32 == 0, (
            f"patch_size={patch_size} non è multiplo di 32. "
            "Questo causerebbe errori di dimensione nelle skip connections della U-Net."
        )

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size
        self.mode = mode
        self.augment = augment and (mode == "train")

        # Raccolta e ordinamento dei file (per riproducibilità)
        self.image_paths: List[Path] = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in extensions]
        )

        # Controlla per tutte le immagini se esiste la corrispondente maschera
        self.valid_pairs: List[Tuple[Path, Path]] = []
        missing = 0
        for img_path in self.image_paths:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                self.valid_pairs.append((img_path, mask_path))
            else:
                missing += 1

        if missing > 0:
            print(f"[Dataset] ATTENZIONE: {missing} immagini senza maschera corrispondente (skippate).")

        print(
            f"[Dataset] Modalità='{mode}' | Coppie valide={len(self.valid_pairs)} | "
            f"patch_size={patch_size}x{patch_size} | Augmentation={self.augment}"
        )

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.valid_pairs[idx]

        # --- Caricamento ---
        # cv2 legge BGR per default -> convertiamo in RGB
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # La maschera è in grayscale: pixel 255=vaso, 0=background
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Impossibile caricare l'immagine: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Impossibile caricare la maschera: {mask_path}")

        # --- Estrazione della patch (solo in training) ---
        if self.mode == "train":
            image, mask = self._extract_random_patch(image, mask)

        # --- Augmentation geometrica (solo in training) ---
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # --- Conversione in tensori PyTorch ---
        image = self._image_to_tensor(image)    # -> (3, H, W), float32, [0, 1]
        mask  = self._mask_to_tensor(mask)      # -> (1, H, W), float32, {0.0, 1.0}

        return image, mask

    # --------------------------------------------------------------------------
    # METODI PRIVATI
    # --------------------------------------------------------------------------

    def _extract_random_patch(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estrae una patch quadrata di dimensione self.patch_size da una posizione
        (r, c) campionata uniformemente nell'immagine.

        La stessa trasformazione geometrica (identico r, c) viene applicata
        sia all'immagine che alla maschera per preservare la corrispondenza spaziale.
        """
        H, W = image.shape[:2]
        p = self.patch_size

        # Campionamento casuale del corner in alto a sinistra
        r = random.randint(0, H - p)
        c = random.randint(0, W - p)

        img_patch  = image[r:r + p, c:c + p]
        mask_patch = mask[r:r + p, c:c + p]

        return img_patch, mask_patch

    def _apply_augmentation(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augmentation geometrica random. Le stesse trasformazioni vengono applicate
        identicamente a immagine e maschera (stesso seed per le operazioni casuali).

        Trasformazioni implementate:
        - Flip orizzontale (p=0.5)
        - Flip verticale (p=0.5)
        - Rotazione 90°, 180°, 270° casuale (p=0.5)
        """
        # Flip orizzontale
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask  = cv2.flip(mask, 1)

        # Flip verticale
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask  = cv2.flip(mask, 0)

        # Rotazione multipla di 90°
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])  # 90°, 180°, 270°
            image = np.rot90(image, k).copy()
            mask  = np.rot90(mask, k).copy()

        return image, mask

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Converte un array NumPy (H, W, C) uint8 in un tensore PyTorch (C, H, W) float32
        normalizzato nell'intervallo [0.0, 1.0].
        """
        image = image.astype(np.float32) / 255.0
        # Da (H, W, C) a (C, H, W) -> formato atteso da PyTorch/U-Net
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image)

    def _mask_to_tensor(self, mask: np.ndarray) -> torch.Tensor:
        """
        Converte la maschera in un tensore binario float32 (1, H, W) con valori in {0.0, 1.0}.
        La soglia a 127 gestisce eventuali artefatti di compressione .png (valori intermedi).
        """
        binary_mask = (mask > 127).astype(np.float32)
        # Aggiunge la dimensione canale: (H, W) -> (1, H, W)
        binary_mask = np.expand_dims(binary_mask, axis=0)
        return torch.from_numpy(binary_mask)


# ==============================================================================
# FACTORY FUNCTION PER I DATALOADER
# ==============================================================================

def get_dataloaders(
    dataset_root: Path,
    patch_size: int = PATCH_SIZE,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Costruisce e restituisce i DataLoader di train, validation e test.

    Il set di training viene ulteriormente diviso in train/validation con
    proporzione (1 - val_split) / val_split.

    Args:
        dataset_root : Path radice del dataset Kaggle (cartella contenente 'train' e 'test').
        patch_size   : Dimensione delle patch per il training.
        batch_size   : Numero di patch per batch nel DataLoader di training.
        val_split    : Frazione del set di training da riservare alla validazione.
        num_workers  : Worker paralleli per il DataLoader (0 = sincrono, più stabile su Windows).
        seed         : Seed per la riproducibilità dello split.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_img_dir  = dataset_root / "train" / "Original"
    train_mask_dir = dataset_root / "train" / "Ground truth"
    test_img_dir   = dataset_root / "test"  / "Original"
    test_mask_dir  = dataset_root / "test"  / "Ground truth"

    # Dataset completo di training (con augmentation)
    full_train_dataset = VesselDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        patch_size=patch_size,
        mode="train",
        augment=True,
    )

    # Split train/validation
    n_total = len(full_train_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [n_train, n_val], generator=generator
    )

    # Dataset di test (nessun patch, nessuna augmentation)
    test_dataset = VesselDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        patch_size=patch_size,
        mode="test",
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"[DataLoader] Train={n_train} | Val={n_val} | Test={len(test_dataset)} immagini\n"
        f"[DataLoader] batch_size={batch_size} (train) | 1 (val/test)"
    )

    return train_loader, val_loader, test_loader
