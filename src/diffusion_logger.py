"""
src/diffusion_logger.py
-----------------------
Sistema di logging per l'inferenza e analisi della diffusion network.

Genera due file di output per ogni run:
    - diff_run_YYYYMMDD_HHMM.log         : Riepilogo con timing, iperparametri e statistiche di inferenza.
    - diff_run_YYYYMMDD_HHMM_images.csv  : Tabella immagine-per-immagine con le metriche (per i plot).

Uso tipico in analyze_diffusion.py:
    logger = DiffusionLogger(log_dir=OUTPUT_DIR / "logs", hparams={...})
    logger.start()
    for img_path in images:
        ...
        logger.log_image(img_name, dice, iou, img_time)
    logger.finish(avg_dice, avg_iou)
"""

import csv
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional


class DiffusionLogger:
    """
    Gestisce il logging completo di un run di inferenza Diffusion.

    Attributi pubblici impostati dopo start():
        run_id  : stringa timestamp 'YYYYMMDD_HHMMSS' univoca per il run.
        log_path: Path al file .log di riepilogo.
        csv_path: Path al file _images.csv con le metriche per immagine.
    """

    # Intestazione del CSV
    _CSV_HEADER = [
        "image_name",
        "dice", "iou",
        "inference_time_s",
    ]

    def __init__(self, log_dir: Path, hparams: dict) -> None:
        """
        Args:
            log_dir : Directory in cui salvare i file di output.
            hparams : Dizionario degli iperparametri da registrare nel .log.
                      Chiavi attese (tutte opzionali, ma consigliate):
                        patch_size, stride, num_timesteps, base_dim, use_ddim, device
        """
        self._log_dir = Path(log_dir)
        self._hparams = hparams

        self._start_dt: Optional[datetime] = None
        self._end_dt:   Optional[datetime] = None

        # Accumula i dati immagine per immagine
        self._image_rows: list[dict] = []

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
        self.log_path = self._log_dir / f"diff_run_{self.run_id}.log"
        self.csv_path = self._log_dir / f"diff_run_{self.run_id}_images.csv"

        # Crea il CSV con solo l'header
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_HEADER)
            writer.writeheader()

        print(f"[Logger] Run avviato  : {self._start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Logger] Log file     : {self.log_path}")
        print(f"[Logger] CSV file     : {self.csv_path}")

    def log_image(
        self,
        image_name:       str,
        dice:             float,
        iou:              float,
        inference_time_s: float,
    ) -> None:
        """
        Registra le metriche di una singola immagine e le appende al CSV in tempo reale.

        Args:
            image_name       : Nome dell'immagine processata.
            dice             : Dice Score dell'immagine.
            iou              : IoU Score dell'immagine.
            inference_time_s : Tempo impiegato per l'inferenza di questa immagine in secondi.
        """
        row = {
            "image_name":       image_name,
            "dice":             round(dice, 6),
            "iou":              round(iou, 6),
            "inference_time_s": round(inference_time_s, 2),
        }
        self._image_rows.append(row)

        # Append immediato al CSV (utile se l'inferenza si interrompe prima della fine)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_HEADER)
            writer.writerow(row)

    def finish(self, avg_dice: float, avg_iou: float) -> None:
        """
        Registra l'ora di fine e scrive il file .log di riepilogo completo.

        Args:
            avg_dice : Dice Score medio finale sul set di test.
            avg_iou  : IoU Score medio finale sul set di test.
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

        # Statistiche globali sui tempi di inferenza
        all_times = [r["inference_time_s"] for r in self._image_rows]
        mean_time = statistics.mean(all_times) if all_times else 0.0
        std_time  = statistics.stdev(all_times) if len(all_times) > 1 else 0.0

        lines = self._build_log(duration_str, mean_time, std_time, avg_dice, avg_iou)

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
        duration_str: str,
        mean_time:    float,
        std_time:     float,
        avg_dice:     float,
        avg_iou:      float,
    ) -> list[str]:
        """Costruisce il contenuto del file .log come lista di stringhe."""
        W = 56  # larghezza del separatore
        SEP  = "=" * W
        DASH = "-" * W
        hp   = self._hparams

        lines = [
            SEP,
            "  VESSEL SEGMENTATION - DIFFUSION INFERENCE LOG",
            SEP,
            f"  Run Start : {self._start_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Run End   : {self._end_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Duration  : {duration_str}",
            "",
            DASH,
            "  HYPERPARAMETERS",
            DASH,
            f"  Patch Size      : {hp.get('patch_size', 'N/A')}",
            f"  Stride          : {hp.get('stride', 'N/A')}",
            f"  Num Timesteps   : {hp.get('num_timesteps', 'N/A')}",
            f"  Base Dim        : {hp.get('base_dim', 'N/A')}",
            f"  Use DDIM        : {hp.get('use_ddim', 'N/A')}",
            f"  Device          : {str(hp.get('device', 'N/A')).upper()}",
            "",
            DASH,
            "  INFERENCE TIMING (media su tutte le immagini)",
            DASH,
            f"  Mean image time : {mean_time:.2f} s  ±  {std_time:.2f} s",
            "",
            DASH,
            "  FINAL SUMMARY (per dettagli vedi il CSV)",
            DASH,
            f"  Avg Test Dice   : {avg_dice:.4f}",
            f"  Avg Test IoU    : {avg_iou:.4f}",
            f"  CSV metriche    : {self.csv_path.name}",
            SEP,
        ]
        return lines
