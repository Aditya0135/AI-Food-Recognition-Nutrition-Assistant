from AI_Food_Recognition_Nutrition_Assistant.config.configuration import ConfigurationManager
from AI_Food_Recognition_Nutrition_Assistant.components.prepare_base_model import PrepareBaseModel
from AI_Food_Recognition_Nutrition_Assistant import logger

STAGE_NAME = "Prepare Base Model"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        preparer = PrepareBaseModel(prepare_base_model_config)
        model = preparer.prepare()
        self.model = model
        return model
    def get_base_model(self):
        return self.model
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e
    
