import os
import urllib.request as request 
from src.Chicken_Disease_Classification.utils.logger import logger
from src.Chicken_Disease_Classification.utils.common import get_size
from src.Chicken_Disease_Classification.entity.config_entity import DataIngestionConfig

import os
from pathlib import Path
import py7zr
from urllib import request

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_path):
            filename, _ = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_path
            )
            logger.info(f" Downloaded file: {filename} | Size: {get_size(Path(filename))}")
        else:
            logger.info(f" File already exists at: {self.config.local_data_path}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with py7zr.SevenZipFile(self.config.local_data_path, mode='r') as archive:
            archive.extractall(path=unzip_path)

        logger.info(f" Extracted 7z archive to: {unzip_path}")