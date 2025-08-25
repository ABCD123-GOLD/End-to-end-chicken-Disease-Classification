import os
import json
import joblib
import base64
import yaml
from pathlib import Path
from typing import Any, List
from box import ConfigBox
from box.exceptions import BoxValueError
from typeguard import typechecked
from src.Chicken_Disease_Classification.utils.logger import logger

@typechecked
def read_yaml(path: Path) -> ConfigBox:
    """
    Read a YAML file and return its contents as a ConfigBox.

    Args:
        path (Path): Path to the YAML file.

    Returns:
        ConfigBox: Parsed YAML data.
    """
    try:
        with open(path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                logger.warning(f"YAML file at {path} is empty. Returning empty ConfigBox.")
                return ConfigBox({})
            return ConfigBox(content)
    except BoxValueError as e:
        raise ValueError(f"YAML file is invalid: {e}")
    except Exception as e:
        raise e


@typechecked
def create_directories(paths: List[Path], verbose: bool = True) -> None:
    """
    Create directories from a list of paths.

    Args:
        paths (List[Path]): List of directory paths to create.
        verbose (bool): Whether to log creation messages.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")

@typechecked
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

@typechecked
def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved successfully at: {path}")

@typechecked
def save_bin(path: Path, data: Any) -> None:
    joblib.dump(data, path)
    logger.info(f"Binary file saved successfully at: {path}")

@typechecked
def load_bin(path: Path) -> Any:
    data = joblib.load(path)
    logger.info(f"Binary file loaded successfully from: {path}")
    return data

@typechecked
def get_size(path: Path) -> str:
    size = round(os.path.getsize(path) / 1024)
    logger.info(f"Size of file {path}: {size} KB")
    return f"{size} KB"

@typechecked
def decodeImage(data: str, filename: Path) -> None:
    with open(filename, "wb") as f:
        f.write(base64.b64decode(data))
    logger.info(f"Image decoded and saved at: {filename}")

@typechecked
def encodeImageIntoBase64(filename: Path) -> str:
    with open(filename, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    logger.info(f"Image encoded into base64 from: {filename}")
    return encoded