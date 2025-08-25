from src.Chicken_Disease_Classification.utils.logger import logger

from src.Chicken_Disease_Classification.pipeline.stage01_data_ingestion import DataIngestionTraningPipline


STAGE_NAME = "Data Ingestion stage"

try:
             
     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
     obj = DataIngestionTraningPipline()
     obj.main()
     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e: 
    logger.exception (e)
    raise e