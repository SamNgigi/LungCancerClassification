from cnnClassifier import logger
from cnnClassifier.utils.common import load_env_variables_from_yaml
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prep_baseline_model import PrepareBaseModelTrainingPipeling
from cnnClassifier.pipeline.stage_03_model_taining import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"{'>'*5} STAGE: {STAGE_NAME} started... {'<'*5}\n")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> STAGE: {STAGE_NAME} completed!! <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare base model"
try:
    logger.info(f"{'*'*20}")
    logger.info(f"{'>'*5} STAGE: {STAGE_NAME} started... {'<'*5}\n")
    obj = PrepareBaseModelTrainingPipeling()
    obj.main()
    logger.info(
        f"\n\n{'>'*5} STAGE: {STAGE_NAME} completed!! {'<'*5}\n\nx{'='*10}x"
    )
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training"
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


STAGE_NAME = "Evaluation Stage"

try:
    logger.info(f"{'*'*20}")
    logger.info(f"{'>'*5} STAGE: {STAGE_NAME} started... {'<'*5}\n")
    load_env_variables_from_yaml("env/env.yaml")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(
        f"\n\n{'>'*5} STAGE: {STAGE_NAME} completed!! {'<'*5}\n\nx{'='*10}x"
    )
except Exception as e:
    raise e