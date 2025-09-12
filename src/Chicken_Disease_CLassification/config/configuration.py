from src.Chicken_Disease_Classification.constant import *
from src.Chicken_Disease_Classification.utils.common import read_yaml, create_directories 
from src.Chicken_Disease_Classification.entity.config_entity import DataIngestionConfig
from src.Chicken_Disease_Classification.entity.config_entity import PrepareBasicModelConfig



class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        #self.params = read_yaml(params_filepath)

        create_directories([Path(self.config["artifacts_root"])])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]

        # Create the directory where the ZIP will be downloaded and extracted
        create_directories([
            Path(config["local_data_path"]).parent,
            Path(config["unzip_dir"])
        ])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(self.config["artifacts_root"]),
            source_URL=config["source_URL"],
            local_data_path=Path(config["local_data_path"]),
            unzip_dir=Path(config["unzip_dir"])
        )
        return data_ingestion_config
    
    #Base model configuration and Ingestion
    
    def get_prepare_base_model_config(self) -> PrepareBasicModelConfig:
        config = self.config.prepare_base_model

        # Create the directory where the model will be stored
        create_directories([
            Path(config.root_dir)
        ])

        prepare_base_model_config = PrepareBasicModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    

