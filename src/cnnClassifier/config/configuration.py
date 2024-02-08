import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfigEntity,
                                                BaseModelConfigEntity,
                                                TrainingConfigEntity,
                                                EvaluationConfigEntity)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath  = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfigEntity:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfigEntity(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_base_model_config(self) -> BaseModelConfigEntity:
        config = self.config.base_model

        create_directories([config.root_dir])

        base_model_config = BaseModelConfigEntity(
            root_dir=Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size = self.params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE,
            params_include_top = self.params.INCLUDE_TOP,
            params_weights = self.params.WEIGHTS,
            params_classes = self.params.CLASSES
        )

        return base_model_config

    def get_training_config(self)->TrainingConfigEntity:
        training = self.config.training
        base_model = self.config.base_model
        params = self.params
        training_data_path = os.path.join(self.config.data_ingestion.unzip_dir, "Chest-CT-Scan-data")

        create_directories([Path(training.root_dir)])

        training_config = TrainingConfigEntity(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_model_base_path=Path(base_model.updated_base_model_path),
            training_data_path=Path(training_data_path),
            params_epoch=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfigEntity:
        eval_config = EvaluationConfigEntity(
            model_path=Path(f"{self.config.training.root_dir}/model.keras"),
            training_data_path=Path(f"{self.config.data_ingestion.unzip_dir}/Chest-CT-Scan-data"),
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI"),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
