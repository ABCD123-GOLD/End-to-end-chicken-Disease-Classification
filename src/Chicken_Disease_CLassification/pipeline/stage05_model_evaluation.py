
from Chicken_Disease_Classification.utils.logger import logger
from Chicken_Disease_Classification.entity.config_entity import DataIngestionConfig
from Chicken_Disease_Classification.config.configuration import ConfigurationManager
from Chicken_Disease_Classification.components.data_ingestion import DataIngestion
from Chicken_Disease_Classification.entity.config_entity import EvaluationConfig
from Chicken_Disease_Classification.components.model_evaluation import Evaluation

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipline:
    def __init__(self):
        pass
    def main(self):
            config = ConfigurationManager()
            val_config = config.get_validation_config()
            evaluation = Evaluation(val_config)
            evaluation.run_evaluation()
            evaluation.save_score()


if __name__ == "__main__":
     try:
            logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
            obj = ModelEvaluationPipline()
            obj.main()
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
     except Exception as e:
        logger.exception (e)
        raise e
        
         