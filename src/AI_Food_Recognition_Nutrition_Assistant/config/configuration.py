from AI_Food_Recognition_Nutrition_Assistant.constants import *
from AI_Food_Recognition_Nutrition_Assistant.utils.common import read_yaml,create_directories
from AI_Food_Recognition_Nutrition_Assistant.entity.config_entity import *
import os

class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH
                 ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
        root_dir = config.root_dir,
        source_url = config.source_url,
        local_data_file = config.local_data_file,
        unzip_dir = config.unzip_dir,   
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        p = self.params
        create_directories([config.root_dir])
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            unzip_dir=Path(config.unzip_dir),
            train_dir=Path(config.train_dir),
            test_dir=Path(config.test_dir),
            splits_dir=Path(config.splits_dir),
            class_names_path=Path(config.class_names_path),
            input_size=p.model.input_size,
            resize_size=p.model.resize_size,
            seed = p.training.seed,
            randaugment_num_ops=p.augmentation.randaugment_num_ops,
            randaugment_magnitude=p.augmentation.randaugment_magnitude,
            random_erasing_p=p.augmentation.random_erasing_p,
            batch_size= p.training.batch_size,
            num_workers=p.training.num_workers,
        )

        return data_preprocessing_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        p = self.params
        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            architecture=p.model.architecture,
            pretrained_weights=p.model.pretrained_weights,
            num_classes=p.training.num_classes,
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        p = self.params
        create_directories([config.root_dir])
        create_directories([config.checkpoint_dir])
        train_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            checkpoint_dir=Path(config.checkpoint_dir),
            base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            num_classes=p.training.num_classes,
            batch_size=p.training.batch_size,
            num_workers=p.training.num_workers,
            epochs=p.training.epochs,
            seed=p.training.seed,
            lr_backbone=p.training.lr_backbone,
            lr_head=p.training.lr_head,
            weight_decay=p.training.weight_decay,
            max_lr=p.training.max_lr,
            pct_start=p.training.pct_start,
            div_factor=p.training.div_factor,
            final_div_factor=p.training.final_div_factor,
            label_smoothing=p.training.label_smoothing,
            patience=p.training.patience,
            use_mixup=p.augmentation.use_mixup,
            use_cutmix=p.augmentation.use_cutmix,
            mixup_alpha=p.augmentation.mixup_alpha,
            cutmix_alpha=p.augmentation.cutmix_alpha,
            mixup_cutmix_prob=p.augmentation.mixup_cutmix_prob,
            use_ema=p.ema.use_ema,
            ema_decay=p.ema.ema_decay,
        )

        return train_config

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        p = self.params
        create_directories([config.root_dir])
        return EvaluationConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(self.config.training.trained_model_path),
            metrics_path=Path(config.metrics_path),
            report_path=Path(config.report_path),
            confusion_matrix_path=Path(config.confusion_matrix_path),
            num_classes=p.training.num_classes,
            batch_size=p.training.batch_size,
            num_workers=p.training.num_workers,
        )

    def get_model_pusher_config(self) -> ModelPusherConfig:
        config = self.config.model_pusher
        create_directories([config.root_dir])
        create_directories([config.model_bundle_path])
        return ModelPusherConfig(
            root_dir=Path(config.root_dir),
            model_bundle_path=Path(config.model_bundle_path),
            model_path=Path(self.config.training.checkpoint_dir) / "best_model.pth",
            class_names_path=Path(self.config.data_preprocessing.class_names_path),
            hf_repo_id=os.getenv("HF_MODEL_REPO_ID", "").strip(),
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            private_repo=(os.getenv("HF_PRIVATE_REPO", "false").strip().lower() in {"1", "true", "yes", "on"}),
            model_filename=os.getenv("HF_MODEL_FILENAME", "best_model.pth"),
            class_names_filename=os.getenv("HF_CLASS_NAMES_FILENAME", "class_names.json"),
        )