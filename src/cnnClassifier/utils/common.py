import os
# * Box is designed to be an easy drop in transparently replacements for dictionaries that adds dot notation access and a load of other features
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
# * ensure is a set of simple assertion helpers that let you write more expressive, literate, concise, and readable Pythonic code for validating conditions
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """#* read yaml file and return ConfigBox

    #* Args:
        #* path_to_yaml (str): path like input
    
    #* Raises:
        #* ValueError: if yaml file is empty
        #* e: empty file 
    
    #* Returns:
        #* ConfigBox: ConfigBox type
    """
    
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded sucessfully!")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True)->None:
    """#* create list of directories

    #* Args:
        #* path_to_directories (list): list of path of directories
        #* ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to True
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict)->None:
    """#* save json data

    #* Args:
       #* path (Path): path to json file
       #* data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path)->ConfigBox:
    """#* load json file data 

    #* Args:
        #* path (Path): path to json file

    #* Returns:
        #* ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f);
        
        logger.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path)->None:
    """#* Save binary file

    #* Args:
       #* data (Any): data to be saved as binary
       #* path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path)->Any:
    """#* Load binary data

    #* Args:
       #* path (Path): path to binary file

    #* Returns:
        #* Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path)->str:
    """#* get size in KB

    #* Args:
       #* path (Path): path of the file

    #* Returns:
       #* str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

def decodeImage(img_string:str, file_name:str)->None:
    img_data =base64.b64decode(img_string)
    with open(file_name, 'wb') as f:
        f.write(img_data)
        f.close()

def encodeImageIntoBase64(croppedImagePath:str)->bytes:
    with open(croppedImagePath, 'rb') as f:
        return base64.b64decode(f.read())
