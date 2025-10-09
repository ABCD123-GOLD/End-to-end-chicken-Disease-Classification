from Chicken_Disease_Classification.utils.logger import logger
from Chicken_Disease_Classification.entity.config_entity import PrepareCallbacksConfig
from Chicken_Disease_Classification.config.configuration import ConfigurationManager
from Chicken_Disease_Classification.components.prepare_base_model import PrepareCallback
from Chicken_Disease_Classification.components.model_traning import Training
import tensorflow as tf 



STAGE_NAME = "Model Training Stage"

class ModelTrainingPipline:
    def __init__(self):
        pass
    
    def main(self):
        # Load configs
        config = ConfigurationManager()
        training_config = config.get_training_config()
        prepare_callbacks_config = config.get_prepare_callbacks_config()

        # Prepare callbacks (e.g., TensorBoard, EarlyStopping)
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        # Initialize Training
        training = Training(config=training_config)

        # Load and recompile base model
        training.get_base_model()

        # Prepare generators
        training.train_valid_generator()

        # Train and save best model (.keras format)
        training.train(callback_list=callback_list)

if __name__ == "__main__":
     try:
            logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
            obj = ModelTrainingPipline()
            obj.main()
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
     except Exception as e:
        logger.exception (e)
        raise e
        