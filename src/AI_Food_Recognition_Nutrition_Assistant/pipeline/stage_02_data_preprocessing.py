from AI_Food_Recognition_Nutrition_Assistant.config.configuration import ConfigurationManager
from AI_Food_Recognition_Nutrition_Assistant.components.data_preprocessing import DataPreprocessing
from AI_Food_Recognition_Nutrition_Assistant import logger

STAGE_NAME = "Data Preprocessing"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        preprocessing_config = config.get_data_preprocessing_config()
        preprocessing = DataPreprocessing(config=preprocessing_config)
        train_loader,val_loader,test_loader = preprocessing.load_and_split()
        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e