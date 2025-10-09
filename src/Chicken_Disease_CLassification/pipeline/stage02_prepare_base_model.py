from Chicken_Disease_Classification.utils.logger import logger
from Chicken_Disease_Classification.entity.config_entity import PrepareBasicModelConfig
from Chicken_Disease_Classification.config.configuration import ConfigurationManager
from Chicken_Disease_Classification.components.prepare_base_model import PrepareBaseModel
import tensorflow as tf


STAGE_NAME = "Prepare Base Model stage"

class PrepareBaseModelTraningPipline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

if __name__ == "__main__":
     try:
            logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
            obj = PrepareBaseModelTraningPipline()
            obj.main()
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
     except Exception as e:
        logger.exception (e)
        raise e
        