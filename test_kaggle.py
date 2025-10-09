# test_data_ingestion.py
from Chicken_Disease_Classification.components.data_ingestion import DataIngestion
from Chicken_Disease_Classification.config.configuration import ConfigurationManager

def test_download():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_from_kagglehub()

if __name__ == "__main__":
    test_download()