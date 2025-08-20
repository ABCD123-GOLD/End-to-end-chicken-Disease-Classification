import os
import json
import joblib
import base64
import yaml
from pathlib import Path
from typing import Any
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from src.Chicken_Disease_Classification import logger

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load a JSON file and return its contents as a ConfigBox.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data as class attributes instead of a dictionary.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        path (Path): Path to save the JSON file.
        data (dict): Data to be saved.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved successfully at: {path}")

@ensure_annotations
def save_bin(path: Path, data: Any) -> None:
    """
    Save binary data using joblib.

    Args:
        path (Path): Path to save the binary file.
        data (Any): Data to be saved.
    """
    joblib.dump(data, path)
    logger.info(f"Binary file saved successfully at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load binary data using joblib.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Loaded data.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded successfully from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get the size of a file in KB.

    Args:
        path (Path): Path to the file.

    Returns:
        str: Size in kilobytes.
    """
    size = round(os.path.getsize(path) / 1024)
    logger.info(f"Size of file {path}: {size} KB")
    return f"{size} KB"

@ensure_annotations
def decodeImage(data: str, filename: Path) -> None:
    """
    Decode a base64 string and save it as an image file.

    Args:
        data (str): Base64 encoded image string.
        filename (Path): Path to save the decoded image.
    """
    with open(filename, "wb") as f:
        f.write(base64.b64decode(data))
    logger.info(f"Image decoded and saved at: {filename}")

@ensure_annotations
def encodeImageIntoBase64(filename: Path) -> str:
    """
    Encode an image file into a base64 string.

    Args:
        filename (Path): Path to the image file.

    Returns:
        str: Base64 encoded string.
    """
    with open(filename, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    logger.info(f"Image encoded into base64 from: {filename}")
    return encoded