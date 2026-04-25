import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from AI_Food_Recognition_Nutrition_Assistant.config.configuration import EvaluationConfig
from AI_Food_Recognition_Nutrition_Assistant.utils.common import get_device, accuracy
from AI_Food_Recognition_Nutrition_Assistant.utils.common import (
    ensure_dir, save_json, save_numpy, load_model
)
from AI_Food_Recognition_Nutrition_Assistant import logger


class ModelEvaluator:
    def __init__(self, config: EvaluationConfig, model: nn.Module,
                 test_loader: DataLoader, class_names: list):
        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.class_names = class_names
        ensure_dir(config.root_dir)

    def load_trained_weights(self) -> None:
        state = load_model(self.config.trained_model_path, self.device)
        self.model.load_state_dict(state)
        logger.info(f"Loaded weights from {self.config.trained_model_path}")

    def evaluate(self) -> dict:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        c1_total, c5_total, n = 0.0, 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):# type: ignore
                    out = self.model(imgs)
                    loss = criterion(out, labels)
                total_loss += loss.item()
                (c1, bs), (c5, _) = accuracy(out, labels, topk=(1, 5))
                c1_total += c1.item()
                c5_total += c5.item()
                n += bs
                all_preds.extend(out.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        top1 = 100.0 * c1_total / n
        top5 = 100.0 * c5_total / n
        avg_loss = total_loss / len(self.test_loader)

        metrics = {
            "top1_accuracy": round(top1, 4),
            "top5_accuracy": round(top5, 4),
            "val_loss": round(avg_loss, 6),
            "num_samples": n,
        }
        logger.info(f"Evaluation -> Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | Loss: {avg_loss:.4f}")

        # Save metrics
        save_json(self.config.metrics_path, metrics)

        # Save classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=self.class_names
        )
        ensure_dir(self.config.report_path.parent)
        with open(self.config.report_path, "w") as f:
            f.write(report)# type: ignore
        logger.info(f"Classification report saved: {self.config.report_path}")

        # Save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        save_numpy(cm, self.config.confusion_matrix_path)
        logger.info(f"Confusion matrix saved: {self.config.confusion_matrix_path}")

        return metrics