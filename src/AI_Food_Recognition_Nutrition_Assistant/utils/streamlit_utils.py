"""
Streamlit utilities for single-image food classification inference.
Handles model loading, image preprocessing, and prediction extraction.
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path

from AI_Food_Recognition_Nutrition_Assistant.config.configuration import ConfigurationManager
from AI_Food_Recognition_Nutrition_Assistant.utils.common import (
    load_model, load_json, get_device
)
from AI_Food_Recognition_Nutrition_Assistant import logger


def load_trained_model_and_config():
    """
    Load trained model and configuration for inference.
    Uses caching to load model once per session.

    Returns:
        tuple: (model, device, class_names, config)
    """
    device = get_device()

    # Load configuration
    config = ConfigurationManager()
    model_config = config.get_training_config()
    data_preprocessing_config = config.get_data_preprocessing_config()

    # Build model architecture
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(
        model.classifier[2].in_features, 101
    )  # type: ignore

    # Load trained weights
    model_path = model_config.checkpoint_dir / "best_model.pth"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"✅ Loaded trained model from {model_path} on {device}")

    # Load class names
    class_names_data = load_json(data_preprocessing_config.class_names_path)
    class_names = class_names_data["class_names"]

    logger.info(f"✅ Loaded {len(class_names)} class names")

    return model, device, class_names, config


def get_val_test_transform(input_size=224, resize_size=256):
    """
    Get validation/test transforms (no augmentation).
    Uses ImageNet normalization statistics.

    Args:
        input_size: Final crop size (default 224)
        resize_size: Intermediate resize size (default 256)

    Returns:
        transforms.Compose: Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def preprocess_image_for_inference(pil_image, input_size=224, resize_size=256):
    """
    Preprocess PIL image for model inference.

    Args:
        pil_image: PIL Image object
        input_size: Target crop size
        resize_size: Intermediate resize size

    Returns:
        torch.Tensor: Preprocessed image with batch dimension (1, 3, 224, 224)
    """
    transform = get_val_test_transform(input_size, resize_size)
    image_tensor = transform(pil_image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def get_top_k_predictions(model, image_tensor, class_names, device, k=5):
    """
    Run inference and extract top-k predictions with confidence scores.

    Args:
        model: PyTorch model (ConvNeXt)
        image_tensor: Preprocessed image tensor (1, 3, 224, 224)
        class_names: List of food class names (101 items)
        device: 'cuda' or 'cpu'
        k: Number of top predictions to return (default 5)

    Returns:
        list: List of tuples (class_name, confidence_pct, index)
              Sorted by confidence in descending order
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Get logits from model
        logits = model(image_tensor)  # Shape: (1, 101)

        # Convert to probabilities using softmax
        probabilities = F.softmax(logits, dim=1)  # Shape: (1, 101)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)

        # Convert to numpy and percentages
        top_probs = (top_probs.cpu().numpy()[0] * 100).round(2)
        top_indices = top_indices.cpu().numpy()[0]

    # Create result list with class names
    predictions = []
    for idx, (prob, class_idx) in enumerate(zip(top_probs, top_indices)):
        class_name = class_names[class_idx]
        predictions.append({
            'rank': idx + 1,
            'class_name': class_name,
            'confidence': float(prob),
            'class_index': int(class_idx)
        })

    logger.info(f"🔮 Top prediction: {predictions[0]['class_name']} ({predictions[0]['confidence']:.2f}%)")

    return predictions


def get_food_emoji(class_name):
    """
    Map food class name to emoji.
    Simple heuristic-based mapping.

    Args:
        class_name: Food class name (e.g., "apple_pie")

    Returns:
        str: Emoji character
    """
    emoji_map = {
        'pizza': '🍕', 'pasta': '🍝', 'bread': '🍞', 'rice': '🍚',
        'steak': '🥩', 'chicken': '🍗', 'fish': '🐟', 'shrimp': '🦐',
        'salad': '🥗', 'soup': '🍲', 'cake': '🍰', 'cookie': '🍪',
        'donut': '🍩', 'ice_cream': '🍦', 'apple': '🍎', 'banana': '🍌',
        'orange': '🍊', 'grapes': '🍇', 'strawberry': '🍓', 'watermelon': '🍉',
        'cheese': '🧀', 'egg': '🥚', 'meat': '🥩', 'hot_dog': '🌭',
        'hamburger': '🍔', 'fries': '🍟', 'taco': '🌮', 'burrito': '🌯',
        'sushi': '🍣', 'dumplings': '🥟', 'noodles': '🍜', 'ramen': '🍜',
    }

    class_lower = class_name.lower()
    for key, emoji in emoji_map.items():
        if key in class_lower:
            return emoji

    return '🍽️'  # Default food plate emoji


def get_food_description(class_name):
    """
    Get simple description for food class.

    Args:
        class_name: Food class name

    Returns:
        str: Simple description
    """
    descriptions = {
        'pizza': 'Italian bread-based dish with cheese and toppings',
        'pasta': 'Italian pasta dish',
        'bread': 'Baked bread product',
        'rice': 'Cooked rice dish',
        'steak': 'Grilled meat cut',
        'chicken': 'Chicken dish',
        'salad': 'Fresh vegetable salad',
        'soup': 'Hot soup',
        'cake': 'Sweet baked dessert',
        'sushi': 'Japanese rice and fish dish',
    }

    class_lower = class_name.lower()
    for key, desc in descriptions.items():
        if key in class_lower:
            return desc

    return f"{class_name.replace('_', ' ').title()} dish"
