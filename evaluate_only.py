"""
Evaluate trained model without running full pipeline.
Usage: python evaluate_only.py
"""
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from PIL import Image

from AI_Food_Recognition_Nutrition_Assistant.config.configuration import ConfigurationManager
from AI_Food_Recognition_Nutrition_Assistant.components.model_evaluator import ModelEvaluator
from AI_Food_Recognition_Nutrition_Assistant.utils.common import load_json
from AI_Food_Recognition_Nutrition_Assistant import logger


def custom_loader(path):
    """Custom image loader that skips corrupted files."""
    try:
        # Skip macOS and system files
        if '._' in str(path) or '__MACOSX' in str(path) or '.DS_Store' in str(path):
            return None

        img = Image.open(path)
        return img.convert('RGB')
    except Exception as e:
        logger.warning(f"⚠️ Skipping corrupted image: {path}")
        return None


class FilteredImageFolder(datasets.ImageFolder):
    """ImageFolder that skips corrupted images."""
    def __init__(self, *args, **kwargs):
        kwargs['loader'] = custom_loader
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            # Skip corrupted images by returning next valid index
            logger.warning(f"Skipping corrupted image at index {index}")
            return self.__getitem__((index + 1) % len(self.imgs))



def evaluate_model():
    """Load trained model and evaluate on test set."""

    # ── 1. Load config ────────────────────────────────────────────────────
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion()
    data_preprocessing_config = config.get_data_preprocessing_config()
    model_config = config.get_training_config()
    eval_config = config.get_evaluation_config()

    logger.info("Loaded configuration")

    # ── 2. Load saved class names ─────────────────────────────────────────
    class_names_data = load_json(data_preprocessing_config.class_names_path)
    class_names = class_names_data["class_names"]
    logger.info(f"Loaded {len(class_names)} class names from {data_preprocessing_config.class_names_path}")

    # ── 3. Load test data ─────────────────────────────────────────────────
    val_test_transform = transforms.Compose([
        transforms.Resize(data_preprocessing_config.resize_size),
        transforms.CenterCrop(data_preprocessing_config.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
    ])

    full_dataset = FilteredImageFolder(
        root=f"{data_ingestion_config.unzip_dir}/food-101/food-101/images",
        transform=val_test_transform
    )

    # Load saved test indices
    splits = torch.load(data_preprocessing_config.splits_dir)
    test_indices = splits["test_indices"]
    test_subset = Subset(full_dataset, test_indices)

    test_loader = DataLoader(
        test_subset,
        batch_size=data_preprocessing_config.batch_size,
        shuffle=False,
        num_workers=data_preprocessing_config.num_workers,
        pin_memory=True,
    )
    logger.info(f"Loaded test set with {len(test_subset)} samples")

    # ── 4. Load model ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 101)# type: ignore

    # Load trained weights
    model_path = model_config.checkpoint_dir / "best_model.pth"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    logger.info(f"Loaded trained model from {model_path}")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    evaluator = ModelEvaluator(
        config=eval_config,
        model=model,
        test_loader=test_loader,
        class_names=class_names
    )
    metrics = evaluator.evaluate()

    logger.info(f"\n{'='*50}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    logger.info(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    logger.info(f"Validation Loss: {metrics['val_loss']:.4f}")
    logger.info(f"{'='*50}\n")

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()
