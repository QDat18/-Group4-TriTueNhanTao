import os
import csv
from typing import Dict, Any


class CSVLogger:
    """
    Ghi log quá trình huấn luyện ra file CSV.
    Dùng để vẽ biểu đồ Loss, Accuracy, Learning Rate trong báo cáo.
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self.fieldnames = [
            "epoch",
            "phase",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "embedding_norm",
            "gradient_norm",
            "num_batches",
            "epoch_time_sec",
            "device",
            "gpu_memory_mb",
            "checkpoint_saved"
        ]

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """Ghi một dòng kết quả vào file CSV."""
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)