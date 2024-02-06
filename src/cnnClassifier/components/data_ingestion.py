import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfigEntity

class DataIngestionComponent:
    def __init__(self, config: DataIngestionConfigEntity):
        self.config = config

    def download_file(self) -> None:
        
        """
        #* Fetches data from provided url

        #* Raises:
           #* e: Exception in case download fails
        """

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(f"{self.config.root_dir}", exist_ok=True)
            logger.info(f"DOWNLOADING data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix+file_id, zip_download_dir)
            
            logger.info(f"SUCCESSFULLY, downloaded data from {dataset_url} into file {zip_download_dir}!")
            
        except Exception as e:
            raise e
        
    def extract_zip_file(self)->None:
        """_summary_
        #* zip_File_path: str
        #* Extracts the zip file into the data directory
        #* Function returns None
        """
        
        uzip_path = self.config.unzip_dir
        os.makedirs(uzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(uzip_path)
