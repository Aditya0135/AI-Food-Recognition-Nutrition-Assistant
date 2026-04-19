from multiprocessing import freeze_support

from AI_Food_Recognition_Nutrition_Assistant import logger
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_03_prepare_base_model import PrepareBaseModelPipeline
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_04_train import TrainModelPipeline


def main():
    # STAGE_NAME = "Data Ingestion Stage"
    # try:
    #     logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    #     data_ingestion = DataIngestionPipeline()
    #     data_ingestion.main()
    #     logger.info(f">>>> stage {STAGE_NAME} completed <<<< \n\nx============x")
    # except Exception as e:
    #     logger.exception(e)
    #     raise e

    stage_name = "Data Preprocessing Stage"
    try:
        logger.info(f">>>> stage {stage_name} started <<<<")
        data_preprocessing = DataPreprocessingPipeline()
        train_loader, val_loader, test_loader = data_preprocessing.main()
        logger.info(f">>>> stage {stage_name} completed <<<< \n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e

    stage_name = "Prepare Base Model Stage"
    try:
        logger.info(f">>>> stage {stage_name} started <<<<")
        base_model = PrepareBaseModelPipeline()
        model = base_model.main()
        logger.info(f">>>> stage {stage_name} completed <<<< \n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e

    stage_name = "Train Model Stage"
    try:
        logger.info(f">>>> stage {stage_name} started <<<<")
        train_model = TrainModelPipeline()
        train_model.main(train_loader, val_loader, model)
        logger.info(f">>>> stage {stage_name} completed <<<< \n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    freeze_support()
    main()
