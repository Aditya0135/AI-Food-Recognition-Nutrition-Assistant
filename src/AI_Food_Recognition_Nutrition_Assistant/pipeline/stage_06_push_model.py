from AI_Food_Recognition_Nutrition_Assistant.config.configuration import ConfigurationManager
from AI_Food_Recognition_Nutrition_Assistant.components.model_pusher import ModelPusher
from AI_Food_Recognition_Nutrition_Assistant import logger

STAGE_NAME = "Push Model"


class PushModelPipeline:
    def __init__(self):
        pass

    def main(self) -> str:
        config = ConfigurationManager()
        pusher_config = config.get_model_pusher_config()
        pusher = ModelPusher(config=pusher_config)
        repo_url = pusher.push_to_hf()
        return repo_url


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = PushModelPipeline()
        repo_url = obj.main()
        logger.info(f"Model pushed to: {repo_url}")
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e
