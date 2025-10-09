from Chicken_Disease_Classification.utils.logger import logger
from Chicken_Disease_Classification.entity.config_entity import PrepareCallbacksConfig
from Chicken_Disease_Classification.config.configuration import ConfigurationManager
from Chicken_Disease_Classification.components.prepare_base_model import PrepareCallback
import tensorflow as tf 

STAGE_NAME = "Prepare Callbacks stage"
class PrepareCallbacksTraningPipline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
        return callback_list
    
if __name__ == "__main__":
     try:
            logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
            obj = PrepareCallbacksTraningPipline()
            obj.main()
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
     except Exception as e:
        logger.exception (e)
        raise e