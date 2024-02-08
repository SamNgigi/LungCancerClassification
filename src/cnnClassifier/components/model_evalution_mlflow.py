from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import tensorflow as tf
from cnnClassifier.entity.config_entity import EvaluationConfigEntity
from cnnClassifier.utils.common import save_json


class EvaluationComponent:
    def __init__(self, config: EvaluationConfigEntity):
        self.config = config

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def _valid_gen(self):
        dataGen_kwargs = dict(
            rescale = 1.0/255,
            validation_split = 0.30
        )

        dataFlow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear",
        )

        valid_dataGen = tf.keras.preprocessing.image.ImageDataGenerator(
            **dataGen_kwargs
        )

        self.valid_generator = valid_dataGen.flow_from_directory(
            directory=self.config.training_data_path,
            subset="validation",
            shuffle=False,
            **dataFlow_kwargs
        )

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def evaluate_model(self):
        self.model = self.load_model(self.config.model_path)
        self._valid_gen()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
        
    def log_into_mlfow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "mode", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
