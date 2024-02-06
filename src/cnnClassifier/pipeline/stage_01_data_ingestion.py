from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestionComponent
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestionComponent(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f"{'>'*5} STAGE: {STAGE_NAME} started... {'<'*5}\n")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(
            f"\n\n{'>'*5} STAGE: {STAGE_NAME} completed!! {'<'*5}\n\nx{'='*10}x"
        )
    except Exception as e:
        logger.exception(e)
        raise e
