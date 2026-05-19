"""
src/logger.py
-------------
Sistema di logging per il training run della segmentazione dei vasi sanguigni.

Genera due file di output per ogni run:
    - run_YYYYMMDD_HHMM.log         : Riepilogo con timing, iperparametri e statistiche batch.
    - run_YYYYMMDD_HHMM_epochs.csv  : Tabella epoch-per-epoch con le metriche (per i plot).

Uso tipico in main.py:
    logger = RunLogger(log_dir=OUTPUT_DIR / "logs", hparams={...})
    logger.start()
    for epoch in ...:
        train_loss, train_dice, train_iou, batch_times = train_one_epoch(...)
        val_loss, val_dice, val_iou = validate(...)
        logger.log_epoch(epoch, train_loss, train_dice, train_iou,
                         val_loss, val_dice, val_iou, epoch_duration_s, batch_times)
    logger.finish(best_val_dice, best_epoch)
"""

import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class RunLogger:
    """
    Gestisce il logging completo di un training run.

    Attributi pubblici impostati dopo start():
        run_id  : stringa timestamp 'YYYYMMDD_HHMMSS' univoca per il run.
        log_path: Path al file .log di riepilogo.
        csv_path: Path al file _epochs.csv con le metriche per epoch.
    """

    # Intestazione del CSV
    _CSV_HEADER = [
        "epoch",
        "train_loss", "train_dice", "train_iou", "train_ahd",
        "val_loss",   "val_dice",   "val_iou",   "val_ahd",
        "epoch_time_s",
        "batch_mean_s", "batch_std_s",
    ]

    def __init__(self, log_dir: Path, hparams: dict) -> None:
        """
        Args:
            log_dir : Directory in cui salvare i file di output.
            hparams : Dizionario degli iperparametri da registrare nel .log.
                      Chiavi attese (tutte opzionali, ma consigliate):
                        patch_size, batch_size, epochs, learning_rate,
                        num_workers, device, random_seed,
                        unet_features, loss_alpha, scheduler_patience, scheduler_factor
        """
        self._log_dir = Path(log_dir)
        self._hparams = hparams

        self._start_dt: Optional[datetime] = None
        self._end_dt:   Optional[datetime] = None

        # Accumula i dati epoch per epoch
        self._epoch_rows: list[dict] = []

        # Path impostati in start()
        self.run_id:   str  = ""
        self.log_path: Path = Path()
        self.csv_path: Path = Path()

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Registra l'ora di inizio e crea i file di output (con header CSV)."""
        try:
            from zoneinfo import ZoneInfo
            self._start_dt = datetime.now(tz=ZoneInfo('Europe/Rome'))
        except ImportError:
            self._start_dt = datetime.now()

        self.run_id    = self._start_dt.strftime("%Y%m%d_%H%M")

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self._log_dir / f"run_{self.run_id}.log"
        self.csv_path = self._log_dir / f"run_{self.run_id}_epochs.csv"

        # Crea il CSV con solo l'header
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_HEADER)
            writer.writeheader()

        print(f"[Logger] Run avviato  : {self._start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Logger] Log file     : {self.log_path}")
        print(f"[Logger] CSV file     : {self.csv_path}")

    def log_epoch(
        self,
        epoch:         int,
        train_loss:    float,
        train_dice:    float,
        train_iou:     float,
        train_ahd:     float,
        val_loss:      float,
        val_dice:      float,
        val_iou:       float,
        val_ahd:       float,
        epoch_time_s:  float,
        batch_times:   list[float],
    ) -> None:
        """
        Registra le metriche di una singola epoch e le appende al CSV in tempo reale.

        Args:
            epoch        : Numero dell'epoch (1-indexed).
            train_*      : Loss, Dice e IoU medi sul training set.
            val_*        : Loss, Dice e IoU medi sul validation set.
            epoch_time_s : Durata dell'intera epoch in secondi.
            batch_times  : Lista dei tempi (in secondi) di ogni batch di training.
        """
        batch_mean = statistics.mean(batch_times) if batch_times else 0.0
        batch_std  = statistics.stdev(batch_times) if len(batch_times) > 1 else 0.0

        row = {
            "epoch":         epoch,
            "train_loss":    round(train_loss, 6),
            "train_dice":    round(train_dice, 6),
            "train_iou":     round(train_iou,  6),
            "train_ahd":     round(train_ahd,  6),
            "val_loss":      round(val_loss,   6),
            "val_dice":      round(val_dice,   6),
            "val_iou":       round(val_iou,    6),
            "val_ahd":       round(val_ahd,    6),
            "epoch_time_s":  round(epoch_time_s, 2),
            "batch_mean_s":  round(batch_mean,   4),
            "batch_std_s":   round(batch_std,    4),
        }
        self._epoch_rows.append(row)

        # Append immediato al CSV (utile se il training si interrompe prima della fine)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_HEADER)
            writer.writerow(row)

    def finish(self, best_val_dice: float, best_epoch: int) -> None:
        """
        Registra l'ora di fine e scrive il file .log di riepilogo completo.

        Args:
            best_val_dice : Miglior Dice Score di validazione raggiunto.
            best_epoch    : Epoch in cui è stato raggiunto il miglior Dice.
        """
        try:
            from zoneinfo import ZoneInfo
            self._end_dt = datetime.now(tz=ZoneInfo('Europe/Rome'))
        except ImportError:
            self._end_dt = datetime.now()
            
        duration = self._end_dt - self._start_dt
        total_seconds = int(duration.total_seconds())
        h, rem = divmod(total_seconds, 3600)
        m, s   = divmod(rem, 60)
        duration_str = f"{h:02d}:{m:02d}:{s:02d}"

        # Statistiche globali sui batch (media delle medie per epoch)
        all_batch_means = [r["batch_mean_s"] for r in self._epoch_rows]
        all_batch_stds  = [r["batch_std_s"]  for r in self._epoch_rows]
        global_batch_mean = statistics.mean(all_batch_means) if all_batch_means else 0.0
        global_batch_std  = statistics.mean(all_batch_stds)  if all_batch_stds  else 0.0

        lines = self._build_log(duration_str, global_batch_mean, global_batch_std,
                                best_val_dice, best_epoch)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[Logger] Run terminato: {self._end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Logger] Durata totale: {duration_str}")
        print(f"[Logger] Log salvato  : {self.log_path}")
        print(f"[Logger] CSV salvato  : {self.csv_path}")

    # ------------------------------------------------------------------
    # Metodi privati
    # ------------------------------------------------------------------

    def _build_log(
        self,
        duration_str:      str,
        global_batch_mean: float,
        global_batch_std:  float,
        best_val_dice:     float,
        best_epoch:        int,
    ) -> list[str]:
        """Costruisce il contenuto del file .log come lista di stringhe."""
        W = 56  # larghezza del separatore
        SEP  = "=" * W
        DASH = "-" * W
        hp   = self._hparams

        lines = [
            SEP,
            "  VESSEL SEGMENTATION - RUN LOG",
            SEP,
            f"  Run Start : {self._start_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Run End   : {self._end_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Duration  : {duration_str}",
            "",
            DASH,
            "  HYPERPARAMETERS",
            DASH,
            f"  Patch Size      : {hp.get('patch_size', 'N/A')}",
            f"  Batch Size      : {hp.get('batch_size', 'N/A')}",
            f"  Epochs          : {hp.get('epochs', 'N/A')}",
            f"  Learning Rate   : {hp.get('learning_rate', 'N/A')}",
            f"  Num Workers     : {hp.get('num_workers', 'N/A')}",
            f"  Device          : {str(hp.get('device', 'N/A')).upper()}",
            f"  Random Seed     : {hp.get('random_seed', 'N/A')}",
            f"  UNet Features   : {hp.get('unet_features', 'N/A')}",
            f"  Loss Alpha      : {hp.get('loss_alpha', 'N/A')}",
            f"  Scheduler       : ReduceLROnPlateau "
            f"(patience={hp.get('scheduler_patience', 'N/A')}, "
            f"factor={hp.get('scheduler_factor', 'N/A')})",
            f"  Random seed     : {hp.get('random_seed', 'N/A')}",
            "",
            DASH,
            "  BATCH TIMING (train, media su tutte le epoch)",
            DASH,
            f"  Mean batch time : {global_batch_mean:.4f} s  ±  {global_batch_std:.4f} s",
            "",
            DASH,
            "  EPOCH SUMMARY (per dettagli vedi il CSV)",
            DASH,
            f"  Best Val Dice   : {best_val_dice:.4f}  (epoch {best_epoch})",
            f"  CSV metriche    : {self.csv_path.name}",
            SEP,
        ]
        return lines
