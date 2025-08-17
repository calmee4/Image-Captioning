# logger_utils.py
import csv, os, sys, logging
from datetime import datetime

def init_logger(save_dir, log_name="train.log"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_name)

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    logger.info(f"Logging to {log_path}")
    return logger

class CSVLogger:
    def __init__(self, save_dir, filename="metrics.csv",
                 fieldnames=("epoch","train_loss","val_loss","val_bleu1","val_bleu4")):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, filename)
        self.fieldnames = fieldnames
        header = not os.path.exists(self.path)
        self.f = open(self.path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        if header:
            self.w.writeheader()

    def write(self, **kwargs):
        row = {k: kwargs.get(k, None) for k in self.fieldnames}
        self.w.writerow(row); self.f.flush()

    def close(self):
        self.f.close()
