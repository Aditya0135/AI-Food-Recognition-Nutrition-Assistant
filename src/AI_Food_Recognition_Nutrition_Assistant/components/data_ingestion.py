import zipfile
import urllib.request as request
from AI_Food_Recognition_Nutrition_Assistant import logger
from AI_Food_Recognition_Nutrition_Assistant.utils.common import get_size
from AI_Food_Recognition_Nutrition_Assistant.entity.config_entity import DataIngestionConfig
import os
from pathlib import Path

class DataIngestion:
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_url,
                filename=self.config.local_data_file
                )
            logger.info(f"{filename} downloaded successfully with info {headers}")
        else:
            logger.info(f"file already exist of size: {get_size(Path(self.config.local_data_file))}")


    def extract_zipfile(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)