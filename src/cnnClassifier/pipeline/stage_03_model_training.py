from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        training_config=config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()

        # define evaluation callback
        def eval_fn(model):
            eval_config=config.get_evaluation_config()
            evaluation=Evaluation(eval_config)
            evaluation.model=model
            evaluation._valid_generator()
            evaluation.score=model.evaluate(evaluation.valid_generator)
            evaluation.log_into_mlflow()

        # runs evaluation & training in the same mlflow
        training.train(evaluation_fn=eval_fn)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} ended <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e