from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prep_baseline_model import PrepareBaseModelTrainingPipeling

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> STAGE: {STAGE_NAME} started... <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> STAGE: {STAGE_NAME} completed!! <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare base model"
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
