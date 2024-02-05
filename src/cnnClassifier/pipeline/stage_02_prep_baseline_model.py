from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.base_model import PrepBaselineModel
from cnnClassifier import logger

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeling:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        base_model_config = config.get_base_model_config()
        base_model = PrepBaselineModel(config=base_model_config)
        base_model.get_base_model()
        base_model.update_base_model()


if __name__ == "__main__":
    try:
        logger.info(f"{'>'*5} STAGE: {STAGE_NAME} started... {'<'*5}\n")
        obj = PrepareBaseModelTrainingPipeling()
        obj.main()
        logger.info(
            f"\n\n{'>'*5} STAGE: {STAGE_NAME} completed!! {'<'*5}\n\nx{'='*10}x"
        )
    except Exception as e:
        logger.exception(e)
        raise e
