from AI_Food_Recognition_Nutrition_Assistant.config.configuration import ConfigurationManager
from AI_Food_Recognition_Nutrition_Assistant.components.prepare_callback import PrepareCallback
from AI_Food_Recognition_Nutrition_Assistant import logger

STAGE_NAME = "Prepare Base Model"

class PrepareCallbackPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_callback_config()
        prepare_base_model = PrepareCallback(config=prepare_base_model_config)
        prepare_base_model.get_tb_ckpt_callbacks()

if __name__ == "main":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = PrepareCallbackPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e


