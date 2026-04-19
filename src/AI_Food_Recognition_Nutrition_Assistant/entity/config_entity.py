from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    unzip_dir: Path
    train_dir: Path
    test_dir: Path
    splits_dir: Path
    input_size: int
    resize_size: int
    randaugment_num_ops: int
    randaugment_magnitude: int
    random_erasing_p: float
    seed: int
    batch_size: int
    num_workers: int

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    architecture: str
    pretrained_weights: str
    num_classes: int

@dataclass(frozen=True)
class PrepareCallbackConfig:
    root_dir: Path
    callback_list: list
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    checkpoint_dir: Path
    base_model_path: Path    
    # hyperparams
    num_classes: int
    batch_size: int
    num_workers: int
    epochs: int
    seed: int
    lr_backbone: float
    lr_head: float
    weight_decay: float
    max_lr: float
    pct_start: float
    div_factor: float
    final_div_factor: float
    label_smoothing: float
    patience: int
    # augmentation
    use_mixup: bool
    use_cutmix: bool
    mixup_alpha: float
    cutmix_alpha: float
    mixup_cutmix_prob: float
    # ema
    use_ema: bool
    ema_decay: float