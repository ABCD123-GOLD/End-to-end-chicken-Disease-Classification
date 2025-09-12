from src.Chicken_Disease_Classification.utils.logger import logger
from src.Chicken_Disease_Classification.pipeline.stage01_data_ingestion import DataIngestionTraningPipline
from src.Chicken_Disease_Classification.pipeline.stage02_prepare_base_model import PrepareBaseModelTraningPipline



STAGE_NAME = "Data Ingestion stage"

try:        
     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
     data_ingestion = DataIngestionTraningPipline()
     data_ingestion.main()
     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e: 
    logger.exception (e)
    raise e



STAGE_NAME = "Prepare Base Model stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    prepare_base_model = PrepareBaseModelTraningPipline()
    prepare_base_model.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception (e)
    raise e


