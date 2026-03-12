import os
from box.exceptions import BoxValueError
import yaml
from AI_Food_Recognition_Nutrition_Assistant import logger
import json
import joblib               # use to load ad unload binary file
from ensure import ensure_annotations # (x: type of x) -> return type __ # like typescript
from box import ConfigBox   # allows us to use keys of dict as obj.key instead of obj["key"] -- useful of yaml file as we will use ot of configuration
from pathlib import Path    # is a class which defines data type as Path
from typing import Any      # is a class similar to Path
import base64               # used to encode and decode dat in binary or other forms. Here used for image

@ensure_annotations
def read_yaml(path_to_yaml: Path)->ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {yaml_file} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
        
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data,f,indent=4)
    
    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path:Path)->ConfigBox:
    with open(path) as f:
        content=json.load(f)
    
    logger.info(f"json loaded loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(path: Path, data: Any):
    joblib.dump(data)
    logger.info(f"binary file saved at path: {path}")


@ensure_annotations
def load_bin(path: Path):
     data = joblib.load(path)
     logger.info(f"binary filed loaded from path: {path}")
     return data

def decodeImage(img_string, img_file):
    img_data = base64.b64decode(img_string)
    with open(img_file) as f:
        f.write(img_data)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

@ensure_annotations
def get_size(path: Path)-> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"