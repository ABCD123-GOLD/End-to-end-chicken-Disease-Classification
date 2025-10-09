import sys
import os
from pathlib import Path

project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from Chicken_Disease_Classification.pipeline.stage01_data_ingestion import DataIngestionTraningPipline
from Chicken_Disease_Classification.pipeline.stage02_prepare_base_model import PrepareBaseModelTraningPipline
from Chicken_Disease_Classification.pipeline.stage03_prepare_callbacks import PrepareCallbacksTraningPipline
from Chicken_Disease_Classification.utils.logger import logger
from Chicken_Disease_Classification.config.configuration import ConfigurationManager
from Chicken_Disease_Classification.components.prepare_callbacks import PrepareCallback
from Chicken_Disease_Classification.components.model_traning import Training
from Chicken_Disease_Classification.components.model_evaluation import Evaluation


def check_stage_completed(stage_name, check_paths):
    """
    Check if a stage has already been completed by verifying output files/folders exist.
    
    Args:
        stage_name: Name of the stage for logging
        check_paths: List of paths to check for existence
    
    Returns:
        Boolean indicating if stage is completed
    """
    all_exist = all(Path(p).exists() for p in check_paths)
    if all_exist:
        logger.info(f">>>>> stage {stage_name} already completed (skipping) <<<<<")
    return all_exist


# Data Ingestion Stage
STAGE_NAME = "Data Ingestion stage"
try:
    # Check if data has already been ingested
    data_paths = [
        "artifacts/data_ingestion/data",
        "artifacts/data_ingestion/extracted"
    ]
    
    if not check_stage_completed(STAGE_NAME, data_paths):
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        data_ingestion = DataIngestionTraningPipline()
        data_ingestion.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# Prepare Base Model Stage
STAGE_NAME = "Prepare Base Model stage"
try:
    # Check if base model has already been prepared
    model_paths = [
        "artifacts/prepare_base_model/base_model.h5",
        "artifacts/prepare_base_model/base_model_updated.h5"
    ]
    
    if not check_stage_completed(STAGE_NAME, model_paths):
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        prepare_base_model = PrepareBaseModelTraningPipline()
        prepare_base_model.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# Prepare Callbacks Stage
STAGE_NAME = "Prepare Callbacks stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    
    config_manager = ConfigurationManager()
    prepare_callbacks_config = config_manager.get_prepare_callbacks_config()
    prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
    callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
    logger.info(f"Callback List: {callback_list}")
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")  
except Exception as e:
    logger.exception(e)
    raise e  


# Model Training Stage
STAGE_NAME = "Model Training Stage"
try:
    # Check if model has already been trained
    trained_model_paths = [
        "artifacts/training/trained_model.keras"
    ]
    
    if not check_stage_completed(STAGE_NAME, trained_model_paths):
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        
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
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")  
except Exception as e:
    logger.exception(e)
    raise e  


# Model Evaluation Stage
STAGE_NAME = "Model Evaluation stage"
try:
    # Check if evaluation has already been completed
    eval_paths = [
        "artifacts/evaluation/score.txt"
    ]
    
    if not check_stage_completed(STAGE_NAME, eval_paths):
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.run_evaluation()
        evaluation.save_score()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    else:
        logger.info(f"Evaluation results already exist at {eval_paths[0]}")
except Exception as e:
    logger.exception(e)
    raise e