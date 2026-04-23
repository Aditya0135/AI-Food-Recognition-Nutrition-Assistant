from AI_Food_Recognition_Nutrition_Assistant.config.configuration import ConfigurationManager
from AI_Food_Recognition_Nutrition_Assistant.components.model_evaluator import ModelEvaluator
from AI_Food_Recognition_Nutrition_Assistant import logger
import torch.nn as nn
STAGE_NAME = "Evaluate Model"

class EvaluateModelPipeline:
    def __init__(self):
        pass

    def main(self, model, test_loader, class_names):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()

        # ── Evaluate ───────────────────────────────────────────────────────
        evaluator = ModelEvaluator(
            config=eval_config,
            model=model,
            test_loader=test_loader,
            class_names=class_names
        )
        metrics = evaluator.evaluate()
        return metrics


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        # obj = EvaluateModelPipeline()
        # obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e