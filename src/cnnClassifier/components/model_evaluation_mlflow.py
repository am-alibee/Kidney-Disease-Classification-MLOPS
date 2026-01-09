import os
import shutil
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier import logger

class Evaluation:
    def __init__(self, config):
        self.config=config

    def _valid_generator(self):
        
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1], 
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_to_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
        
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        # with mlflow.start_run(run_name="evaluation"):
        #     # log params: all hyperparams
        #     mlflow.log_params(self.config.all_params)
        mlflow.log_metrics(
            {"eval_loss": self.score[0], "eval_accuracy": self.score[1]}
        )

        mlflow.keras.log_model(
            self.model,
            artifact_path="eval_model",
            registered_model_name=None
        )

    def deploy_model(self, deploy_folder: str = 'model'):
        """Copy the evaluated model to a deployment folder"""
        os.makedirs(deploy_folder, exist_ok=True)
        deploy_path=os.path.join(deploy_folder, "model.h5")
        shutil.copy(self.config.path_to_model, deploy_path)
        logger.info(f"Model deployed to: {deploy_path}")