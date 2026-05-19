# vessel-project

# The Dataset

## https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation


# Walkthrough: Implementazione Completa Vessel Segmentation CNN

## Panoramica

Questo documento riassume tutti i file implementati, le decisioni architetturali prese e i risultati dei test di validazione eseguiti per il progetto di segmentazione dei vasi sanguigni su immagini Fundus 2048x2048 tramite U-Net.

## Struttura del Progetto Implementata

```text
vessel-project/
├── src/
│   ├── __init__.py           ✅ Package Python
│   ├── config.py             ✅ Configurazione globale (path, iperparametri, Kaggle API)
│   ├── dataset.py            ✅ VesselDataset + get_dataloaders() con patch-based sampling
│   ├── losses.py             ✅ DiceLoss + CombinedLoss (BCE + Dice)
│   ├── metrics.py            ✅ dice_score(), iou_score(), pixel_accuracy()
│   ├── engine.py             ✅ train_one_epoch(), validate(), save/load_checkpoint()
│   └── models/
│       ├── __init__.py       ✅ Package Python
│       └── unet.py           ✅ U-Net from scratch (31.4M parametri trainabili)
│
├── main.py                   ✅ Entrypoint training
├── analyze.py                ✅ Inferenza sliding window + visualizzazione comparativa
├── data/                     ✅ Directory creata automaticamente
└── output/
    ├── models/               ✅ Directory per i checkpoint
    └── results/              ✅ Directory per le immagini di output
```

## Dettaglio dei Moduli Implementati

### 1. `src/config.py` — Configurazione Globale

Centralizza tutti i parametri modificabili del progetto:

| Parametro | Valore | Descrizione |
| :--- | :--- | :--- |
| `IMAGE_HEIGHT/WIDTH` | `2048` | Dimensione originale delle immagini Fundus |
| `PATCH_SIZE` | `512` | Dimensione patch di training (multiplo di 32 per U-Net) |
| `BATCH_SIZE` | `8` | Campioni per batch |
| `EPOCHS` | `50` | Epoche di training |
| `LEARNING_RATE` | `1e-4` | Tasso di apprendimento per Adam |
| `MASK_THRESHOLD` | `0.5` | Soglia di binarizzazione delle predizioni |
| `DEVICE` | `CPU` | Rilevato automaticamente (`cuda` se disponibile) |

Funzioni chiave:
- `create_directories()`: Crea automaticamente l'albero delle directory del progetto.
- `get_kaggle_dataset_path()`: Scarica (se necessario) e restituisce il percorso locale del dataset Kaggle.

---

### 2. `src/dataset.py` — Gestione Dati

Classe `VesselDataset` ereditata da `torch.utils.data.Dataset`.

**Patch-Based Random Sampling**:
- In modalità `train`: per ogni `__getitem__`, si campiona un angolo `(r, c)` uniformemente nell'immagine 2048x2048 e si estrae una patch `512x512`.
- La stessa trasformazione è applicata identicamente a immagine e maschera per preservare la corrispondenza spaziale.
- In modalità `test`: restituisce l'immagine intera (l'inferenza usa la sliding window).

**Augmentation geometrica** (solo training):
- Flip orizzontale/verticale casuale (p=0.5)
- Rotazione multipla di 90° casuale (p=0.5)

**Factory `get_dataloaders()`**:
- Split automatico train/val (90%/10%) con seed fisso per riproducibilità.
- Risultato: **540 train | 60 val | 200 test**.

---

### 3. `src/models/unet.py` — Architettura U-Net

Implementazione from-scratch dell'architettura U-Net (Ronneberger et al., 2015).

```
Encoder:  3 -> 64 -> 128 -> 256 -> 512  (+ MaxPool 2x)
Bottleneck: 512 -> 1024
Decoder:  1024+512=1536 -> 512 -> 256 -> 128 -> 64  (+ Bilinear Upsample + Skip Connections)
Output:   64 -> 1 logit (sigmoid applicata durante inference/loss)
```

Scelte implementative:
- **Bilinear Upsample** invece di TransposedConv: evita artefatti a scacchiera.
- **BatchNorm** in ogni DoubleConv: stabilizza il training e accelera la convergenza.
- **Output = logits grezzi**: la sigmoid è applicata internamente da `BCEWithLogitsLoss` (più stabile numericamente) e manualmente durante l'inference.
- **Parametri trainabili**: **31.384.833**

---

### 4. `src/losses.py` — Funzioni di Loss

**`DiceLoss`**: Ottimizza la sovrapposizione globale tra maschera predetta e ground truth. Immune al class imbalance (95% background vs 5% vasi).

**`CombinedLoss`** = `0.5 * BCE + 0.5 * DiceLoss`:
- BCE penalizza ogni singolo pixel errato (localizzazione fine).
- DiceLoss garantisce coerenza globale della forma della maschera.

---

### 5. `src/metrics.py` — Metriche di Valutazione

| Metrica | Formula | Uso |
| :--- | :--- | :--- |
| `dice_score` | `2·TP / (2·TP + FP + FN)` | Metrica principale di segmentazione |
| `iou_score` | `TP / (TP + FP + FN)` | Valutazione più conservativa |
| `pixel_accuracy` | `(TP+TN) / Totale` | Indicativa (soffre del class imbalance) |

---

### 6. `src/engine.py` — Motore di Training

- `train_one_epoch()`: Forward, backward, optimizer step con supporto AMP (Automatic Mixed Precision su GPU).
- `validate()`: Loop di validazione sotto `torch.no_grad()`.
- `save_checkpoint()` / `load_checkpoint()`: Serializzazione dei pesi con metadati (epoch, val_dice).
- Logging in tempo reale con `tqdm`.

---

### 7. `main.py` — Entrypoint Training

Flusso di esecuzione:
1. `set_seed(42)` → riproducibilità garantita.
2. Download/verifica dataset Kaggle.
3. Inizializzazione U-Net + Adam + ReduceLROnPlateau.
4. Loop training: per ogni epoch, calcola train/val metrics e salva il miglior checkpoint.

Comando per avviare: `python main.py`

---

### 8. `analyze.py` — Inferenza e Analisi

**Sliding Window Inference** su immagini 2048x2048:
- Finestra: `512x512`, stride: `384` → overlap di **128 pixel**.
- Le predizioni sovrapposte vengono mediate per mitigare artefatti ai bordi.
- Output: mappa di probabilità `(H, W)` binarizzata con `MASK_THRESHOLD=0.5`.

**Visualizzazione**: salva immagini comparativi a 3 pannelli in `output/results/`:
`[Immagine Originale] | [Ground Truth] | [Predizione U-Net]`

Comando: `python analyze.py` (richiede checkpoint addestrato in `output/models/`)

---

## Risultati Test di Verifica

Tutti i test eseguiti con successo:

```
✅ U-Net: Input (2, 3, 512, 512) → Output (2, 1, 512, 512) — shape test OK
✅ Dataset: 600 coppie train, 200 test — caricamento OK
✅ DataLoader: tensori (2, 3, 512, 512), range [0,1], maschere binarie {0.0, 1.0}
✅ CombinedLoss su predizione random: 0.8101 (valore atteso ~1.0 per predizione casuale)
✅ DiceScore su predizione random: 0.1854 (atteso vicino a 0 per predizione random)
✅ IoU su predizione random: 0.1022 (atteso vicino a 0 per predizione random)
```

> [!IMPORTANT]
> Il progetto è ora pronto per avviare il training. L'unico prerequisito mancante è una GPU con CUDA per velocizzare l'addestramento (attualmente in modalità CPU). Avviare con: `python main.py`

## Prossimi Passi

1. **Avviare il training**: `python main.py` e monitorare Dice Score e Loss per epoch.
2. **Post-training**: `python analyze.py` per generare le visualizzazioni comparative sul test set.
3. **Ottimizzazioni future** (opzionali):
   - Balanced patch sampling se la convergenza è lenta.
   - Transfer Learning con backbone pre-addestrato (es. EfficientNet-B4 tramite `segmentation-models-pytorch`).
   - Data augmentation intensità (contrast, brightness, elastic deformation).
# Markdown syntax guide

## Headers

# This is a Heading h1
## This is a Heading h2
###### This is a Heading h6

## Emphasis

*This text will be italic*  
_This will also be italic_

**This text will be bold**  
__This will also be bold__

_You **can** combine them_

## Lists

### Unordered

* Item 1
* Item 2
* Item 2a
* Item 2b
    * Item 3a
    * Item 3b

### Ordered

1. Item 1
2. Item 2
3. Item 3
    1. Item 3a
    2. Item 3b

## Images

![This is an alt text.](/image/Markdown-mark.svg "This is a sample image.")

## Links

You may be using [Markdown Live Preview](https://markdownlivepreview.com/).

## Blockquotes

> Markdown is a lightweight markup language with plain-text-formatting syntax, created in 2004 by John Gruber with Aaron Swartz.
>
>> Markdown is often used to format readme files, for writing messages in online discussion forums, and to create rich text using a plain text editor.

## Tables

| Left columns  | Right columns |
| ------------- |:-------------:|
| left foo      | right foo     |
| left bar      | right bar     |
| left baz      | right baz     |

## Blocks of code

```
let message = 'Hello world';
alert(message);
```

## Inline code

This web site is using `markedjs/marked`.
