from AI_Food_Recognition_Nutrition_Assistant.config.configuration import ConfigurationManager
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_03_prepare_base_model import PrepareBaseModelPipeline
from AI_Food_Recognition_Nutrition_Assistant.components.model_trainer import ModelTrainer
from AI_Food_Recognition_Nutrition_Assistant import logger

STAGE_NAME = "Train Model"

class TrainModelPipeline:
    def __init__(self):
        pass

    def main(self,train_loader,val_loader,model):
        config = ConfigurationManager()
        config = config.get_training_config()
        trainer = ModelTrainer(
            config=config,
            model = model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        trainer.train()

if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        logger.info("Run this stage from main.py")
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e


