import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfigEntity

class TrainingComponet:
    def __init__(self, config: TrainingConfigEntity):
        self.config = config
        
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_model_base_path
        )
        
    def train_valid_generator(self):
        dataGen_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.20
        )
        
        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear"
        )
        
        valid_dataGen = tf.keras.preprocessing.image.ImageDataGenerator(
            **dataGen_kwargs
        )
        
        self.valid_generator = valid_dataGen.flow_from_directory(
            directory = self.config.training_data_path,
            subset = "validation",
            shuffle = False,
            **dataflow_kwargs
        )
        
        if(self.config.params_is_augmentation):
            train_dataGen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range = 40,
                horizontal_flip = True,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                shear_range = 0.2,
                zoom_range = 0.2,
                **dataGen_kwargs
            )
        else:
            train_dataGen = valid_dataGen
            
        self.train_generator = train_dataGen.flow_from_directory(
            directory = self.config.training_data_path,
            subset = "training",
            shuffle = True,
            **dataflow_kwargs
        )
        
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        self.model.fit(
            self.train_generator,
            epochs = self.config.params_epoch,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = self.validation_steps,
            validation_data = self.valid_generator
        )
        
        self.save_model(
            path = self.config.trained_model_path,
            model = self.model
        )
        
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)