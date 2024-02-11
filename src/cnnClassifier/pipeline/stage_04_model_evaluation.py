from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evalution_mlflow import EvaluationComponent
from cnnClassifier import logger

STAGE_NAME = "Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        eval_configs = config.get_evaluation_config()
        model_evaluator = EvaluationComponent(config=eval_configs)
        model_evaluator.evaluate_model()
        model_evaluator.save_score()
        # #? COMMENT OUT DURING DEPLOY TO PRODUCTION
        # model_evaluator.log_into_mlfow()


if __name__ == "__main__":
    try:
        logger.info(f"{'*'*20}")
        logger.info(f"{'>'*5} STAGE: {STAGE_NAME} started... {'<'*5}\n")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(
            f"\n\n{'>'*5} STAGE: {STAGE_NAME} completed!! {'<'*5}\n\nx{'='*10}x"
        )
    except Exception as e:
        raise e
