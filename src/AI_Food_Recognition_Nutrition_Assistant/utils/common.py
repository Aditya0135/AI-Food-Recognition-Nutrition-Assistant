import os
from box.exceptions import BoxValueError
import yaml
from AI_Food_Recognition_Nutrition_Assistant import logger
import json
import joblib               # use to load ad unload binary file
from ensure import ensure_annotations # (x: type of x) -> return type __ # like typescript
from box import ConfigBox   # allows us to use keys of dict as obj.key instead of obj["key"] -- useful of yaml file as we will use ot of configuration
from pathlib import Path    # is a class which defines data type as Path
from typing import Any      # is a class similar to Path
import base64               # used to encode and decode dat in binary or other forms. Here used for image
import random
import numpy as np
import torch
from pathlib import Path


@ensure_annotations
def read_yaml(path_to_yaml: Path)->ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {yaml_file} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
        
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data,f,indent=4)
    
    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path:Path)->ConfigBox:
    with open(path) as f:
        content=json.load(f)
    
    logger.info(f"json loaded loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def load_bin(path: Path):
     data = joblib.load(path)
     logger.info(f"binary filed loaded from path: {path}")
     return data

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

@ensure_annotations
def get_size(path: Path)-> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.debug(f"Seed set to {seed}")
 
 
def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    return device
 
 
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute top-k accuracy. Returns list of (correct_count, batch_size) tuples."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k, batch_size))
    return res

def precision_recall_f1(preds, labels, num_classes):
        preds = preds.argmax(dim=1)

        # Initialize counters
        TP = torch.zeros(num_classes, device=preds.device)
        FP = torch.zeros(num_classes, device=preds.device)
        FN = torch.zeros(num_classes, device=preds.device)

        for c in range(num_classes):
            TP[c] = ((preds == c) & (labels == c)).sum()
            FP[c] = ((preds == c) & (labels != c)).sum()
            FN[c] = ((preds != c) & (labels == c)).sum()

        # Avoid division by zero
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Macro average
        return precision.mean().item(), recall.mean().item(), f1.mean().item()