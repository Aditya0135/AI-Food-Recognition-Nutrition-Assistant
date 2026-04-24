import random
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from AI_Food_Recognition_Nutrition_Assistant.config.configuration import TrainingConfig
from AI_Food_Recognition_Nutrition_Assistant.utils.common import create_directories,set_seed, get_device, accuracy, precision_recall_f1
from AI_Food_Recognition_Nutrition_Assistant import logger

# ── EMA ──────────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = "cpu"):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))


# ── MixUp / CutMix ───────────────────────────────────────────────────────────
def _rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = (1.0 - lam) ** 0.5
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def mixup_data(x, y, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    x1, y1, x2, y2 = _rand_bbox(x.size(), lam)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2))
    return x, y, y[idx], lam


def mixed_loss(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


# ── Trainer ───────────────────────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self, config: TrainingConfig, train_loader: DataLoader, val_loader: DataLoader, model: nn.Module):
        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        set_seed(config.seed)
        create_directories([config.root_dir])
        create_directories([config.checkpoint_dir])

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.optimizer = AdamW([
            {"params": self.model.features.parameters(), "lr": config.lr_backbone},# type: ignore
            {"params": self.model.classifier.parameters(), "lr": config.lr_head},# type: ignore
        ], weight_decay=config.weight_decay)

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.max_lr,
            steps_per_epoch=len(train_loader),
            epochs=config.epochs,
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
        )
        self.scaler = torch.amp.GradScaler(device=self.device)# type: ignore
        self.ema = ModelEMA(self.model, config.ema_decay, self.device) if config.use_ema else None

    # ── augmentation dispatch ─────────────────────────────────────────────────
    def _apply_aug(self, x, y):
        if not (self.config.use_mixup or self.config.use_cutmix):
            return x, y, y, 1.0, False
        if random.random() >= self.config.mixup_cutmix_prob:
            return x, y, y, 1.0, False

        if self.config.use_mixup and self.config.use_cutmix:
            fn = mixup_data if random.random() < 0.5 else cutmix_data
        elif self.config.use_mixup:
            fn = mixup_data
        else:
            fn = cutmix_data

        alpha = self.config.mixup_alpha if fn is mixup_data else self.config.cutmix_alpha
        x, ya, yb, lam = fn(x, y, alpha)
        return x, ya, yb, lam, True

    # ── one epoch ─────────────────────────────────────────────────────────────
    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for imgs,labels in tqdm(self.train_loader, desc="Training", leave=False):
             imgs, labels = imgs.to(self.device), labels.to(self.device)
             imgs, ya, yb, lam, aug = self._apply_aug(imgs, labels)
             self.optimizer.zero_grad()
             with torch.amp.autocast(device_type=self.device, dtype=torch.float16):# type: ignore
                 out = self.model(imgs)
                 loss = mixed_loss(self.criterion, out, ya, yb, lam) if aug else self.criterion(out, labels)
             self.scaler.scale(loss).backward()
             self.scaler.step(self.optimizer)
             self.scaler.update()
             self.scheduler.step()
             if self.ema:
                 self.ema.update(self.model)
             total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate(self):
        eval_model = self.ema.ema if (self.ema and self.config.use_ema) else self.model
        eval_model.eval()

        total_loss, c1, c5, n = 0.0, 0.0, 0.0, 0
        total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):# type: ignore
                    out = eval_model(imgs)
                    loss = self.criterion(out, labels)

                total_loss += loss.item()

                # Accuracy
                (c1k, bs), (c5k, _) = accuracy(out, labels, topk=(1, 5))
                c1 += c1k.item()
                c5 += c5k.item()
                n += bs

                # Precision / Recall / F1
                p, r, f1 = precision_recall_f1(out, labels, self.config.num_classes)
                total_precision += p
                total_recall += r
                total_f1 += f1

        # Average over batches
        avg_precision = total_precision / len(self.val_loader)
        avg_recall = total_recall / len(self.val_loader)
        avg_f1 = total_f1 / len(self.val_loader)

        return (
            total_loss / len(self.val_loader),
            100.0 * c1 / n,
            100.0 * c5 / n,
            avg_precision,
            avg_recall,
            avg_f1,
        )


    # ── main train loop ───────────────────────────────────────────────────────
    def train(self) -> str:
        best_acc, wait = 0.0, 0
        best_path = self.config.checkpoint_dir / "best_model.pth"

        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch()
            val_loss, top1, top5, avg_precision, avg_recall, avg_f1 = self._validate()
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Top-1: {top1:.2f}% | Top-5: {top5:.2f}%"
                f" | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}"
            )
            if top1 > best_acc:
                best_acc = top1
                wait = 0
                torch.save(self.model.state_dict(), best_path)
                logger.info(f" New best Top-1: {best_acc:.2f}% — checkpoint saved.")
            else:
                wait += 1
                if wait >= self.config.patience:
                    logger.info("Early stopping triggered.")
                    break

        torch.save(self.model.state_dict(), self.config.trained_model_path)
        logger.info(f"Training complete. Best Top-1: {best_acc:.2f}%")
        return str(self.config.trained_model_path)