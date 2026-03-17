from AI_Food_Recognition_Nutrition_Assistant import logger
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<< \n\nx============x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Base Model Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_ingestion = PrepareBaseModelPipeline()
    data_ingestion.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<< \n\nx============x")
except Exception as e:
    logger.exception(e)
    raise e