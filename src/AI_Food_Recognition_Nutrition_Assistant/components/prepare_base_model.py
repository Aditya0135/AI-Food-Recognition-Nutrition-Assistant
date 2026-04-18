import torch.nn as nn
import torch
from torchvision import models
from AI_Food_Recognition_Nutrition_Assistant import logger
from AI_Food_Recognition_Nutrition_Assistant.config.configuration import PrepareBaseModelConfig
from AI_Food_Recognition_Nutrition_Assistant.utils.common import create_directories


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self) -> nn.Module:
        logger.info(f"Loading {self.config.architecture} with weights={self.config.pretrained_weights}")
        model = models.convnext_tiny(weights=self.config.pretrained_weights)
        create_directories([self.config.root_dir])
        torch.save(model.state_dict(),self.config.base_model_path) 
        return model

    def update_model_head(self, model: nn.Module) -> nn.Module:
        in_features = model.classifier[2].in_features # type: ignore[index]
        model.classifier[2] = nn.Linear(in_features, self.config.num_classes) # type: ignore[index]
        logger.info(f"Updated classifier head: {in_features} -> {self.config.num_classes} classes")
        torch.save(model.state_dict(), self.config.updated_base_model_path)
        return model

    def prepare(self) -> nn.Module:
        model = self.get_base_model()
        model = self.update_model_head(model)
        return model