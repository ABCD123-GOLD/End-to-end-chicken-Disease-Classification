from pathlib import Path
import os
from Chicken_Disease_Classification.constant import *
from Chicken_Disease_Classification.utils.common import read_yaml, create_directories
from Chicken_Disease_Classification.entity.config_entity import DataIngestionConfig
from Chicken_Disease_Classification.entity.config_entity import PrepareBasicModelConfig
from Chicken_Disease_Classification.entity.config_entity import PrepareCallbacksConfig,TrainingConfig,EvaluationConfig

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath) 
        
        create_directories([Path(self.config["artifacts_root"])])

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.get("data_ingestion")
        if config is None:
            raise ValueError("Missing 'data_ingestion' section in config.yaml")

        # Only create directory if local_data_path is provided
        local_path = config.get("local_data_path")
        if local_path:
            create_directories([Path(local_path).parent])

        create_directories([Path(config["unzip_dir"])])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(self.config["artifacts_root"]),
            kaggle_dataset=config["kaggle_dataset"],
            local_data_path=Path(local_path) if local_path else None,
            unzip_dir=Path(config["unzip_dir"])
        )
        return data_ingestion_config

    # Base model configuration and Ingestion
    def get_prepare_base_model_config(self) -> PrepareBasicModelConfig:
        config = self.config["prepare_base_model"]  

        # Create the directory where the model will be stored
        create_directories([
            Path(config["root_dir"])  
        ])

        prepare_base_model_config = PrepareBasicModelConfig(
            root_dir=Path(config["root_dir"]), 
            base_model_path=Path(config["base_model_path"]),  
            updated_base_model_path=Path(config["updated_base_model_path"]),  
            params_image_size=self.params["IMAGE_SIZE"],  
            params_learning_rate=self.params["LEARNING_RATE"],  
            params_include_top=self.params["INCLUDE_TOP"],
            params_weights=self.params["WEIGHTS"], 
            params_classes=self.params["CLASSES"]  
        )

        return prepare_base_model_config
    

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        prepare_callbacks_config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(prepare_callbacks_config.checkpoint_model_filepath)
        create_directories([
            Path(str(prepare_callbacks_config.root_dir)),
            Path(str(model_ckpt_dir)), 
            Path(str(prepare_callbacks_config.tensorboard_root_log_dir))
        ])
        
        prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir=Path(prepare_callbacks_config.root_dir),
            checkpoint_model_filepath=Path(prepare_callbacks_config.checkpoint_model_filepath),
            tensorboard_root_log_dir=Path(prepare_callbacks_config.tensorboard_root_log_dir),
            checkpoint_model_dir=Path(prepare_callbacks_config.checkpoint_model_dir),
            checkpoint_model_filename=Path(prepare_callbacks_config.checkpoint_model_filename)
        )
        
        return prepare_callbacks_config
    
    def get_training_config(self) -> TrainingConfig:

        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            image_data_dir=Path(training.image_data_dir),
            label_csv_path=Path(training.label_csv_path),
            params_image_size=params.IMAGE_SIZE,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_learning_rate=params.LEARNING_RATE
        )
        
        return training_config

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        prepare_callbacks_config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(prepare_callbacks_config.checkpoint_model_filepath)
        create_directories([
            Path(str(prepare_callbacks_config.root_dir)),
            Path(str(model_ckpt_dir)), 
            Path(str(prepare_callbacks_config.tensorboard_root_log_dir))
        ])

        prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir=Path(prepare_callbacks_config.root_dir),
            checkpoint_model_filepath=Path(prepare_callbacks_config.checkpoint_model_filepath),
            tensorboard_root_log_dir=Path(prepare_callbacks_config.tensorboard_root_log_dir),
            checkpoint_model_dir=Path(prepare_callbacks_config.checkpoint_model_dir),
            checkpoint_model_filename=Path(prepare_callbacks_config.checkpoint_model_filename)
        )

        return prepare_callbacks_config
    
    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path("artifacts/training/trained_model_best.keras"),
            label_csv_path=Path(self.config.training.label_csv_path),       
            image_data_dir=Path(self.config.training.image_data_dir),       
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        
        return eval_config