from multiprocessing import freeze_support
import torch
from AI_Food_Recognition_Nutrition_Assistant import logger
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_03_prepare_base_model import PrepareBaseModelPipeline
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_04_train import TrainModelPipeline
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_05_evaluate_model import EvaluateModelPipeline as EvaluateModelPipeline
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_06_push_model import PushModelPipeline
from AI_Food_Recognition_Nutrition_Assistant.config.configuration import ConfigurationManager

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
        class_names,train_loader, val_loader, test_loader = data_preprocessing.main()
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

    stage_name = "Evaluate Model Stage"
    try:
        logger.info(f">>>> stage {stage_name} started <<<<")

        # Load trained weights into model
        config = ConfigurationManager()
        model_config = config.get_training_config()
        model_path = model_config.checkpoint_dir / "best_model.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded trained weights from {model_path}")

        evaluate_model = EvaluateModelPipeline()
        matrices = evaluate_model.main(model=model, test_loader=test_loader, class_names=class_names)
        logger.info(f">>>> stage {stage_name} completed <<<< \n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e

    stage_name = "Push Model Stage"
    try:
        logger.info(f">>>> stage {stage_name} started <<<<")
        push_model = PushModelPipeline()
        repo_url = push_model.main()
        logger.info(f"Model pushed to: {repo_url}")
        logger.info(f">>>> stage {stage_name} completed <<<< \n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
    freeze_support()
    main()
