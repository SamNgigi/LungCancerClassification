from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import TrainingComponet
from cnnClassifier import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        model_trainer = TrainingComponet(config=training_config)
        model_trainer.get_base_model()
        model_trainer.train_valid_generator()
        model_trainer.train()


if __name__ == "__main__":
    try:
        logger.info(f"{'*'*20}")
        logger.info(f"{'>'*5} STAGE: {STAGE_NAME} started... {'<'*5}\n")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(
            f"\n\n{'>'*5} STAGE: {STAGE_NAME} completed!! {'<'*5}\n\nx{'='*10}x"
        )
    except Exception as e:
        logger.exception(e)
        raise e
