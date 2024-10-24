import os
import json
import yaml
from pathlib import Path
from typing import Iterator, List, Dict, Any

def iterate_over_files(folder_path: str | Path) -> Iterator[str]:
    """
    Iterate over all files in a given folder and its subfolders.

    Args:
        folder_path: Path to the folder to iterate over.

    Yields:
        Path to each file.

    Returns:
        Iterator of file paths.
    """
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    
    return (str(file_path) for file_path in folder_path.rglob('*') if file_path.is_file())

def parse_r2r_error(r2r_exc) -> str:
    """
    Parse the R2R exception message.

    Args:
        r2r_exc: The R2R exception object containing the error message.

    Raises: 
        ValueError: If the error message cannot be parsed.

    Returns:
        The parsed error message.
    """
    try:
        parsed_err_msg = json.loads(r2r_exc.message)
        return parsed_err_msg['detail']['message']
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to parse R2R error message: {str(e)}")

def load_prompt(directory_path: Path, prompt_name: str) -> Dict[str, Any]:
    """
    Load a prompt from YAML files in a given directory.

    Args:
        directory_path: Path to the directory containing the YAML files.
        prompt_name: Name of the prompt to load.

    Raises:
        ValueError: If the directory path is invalid or prompt is not found.

    Returns:
        The loaded prompt data containing template and input_types.
    """
    if not directory_path or not directory_path.is_dir():
        raise ValueError(f"Invalid directory path: {directory_path}")

    for yaml_file in directory_path.glob("*.yaml"):
        try:
            with yaml_file.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
                
                if not data:
                    continue
                    
                if not isinstance(data, dict):
                    continue
                
                if prompt_data := data.get(prompt_name):
                    return prompt_data
                    
        except (yaml.YAMLError, KeyError) as e:
            # Log error but continue checking other files
            print(f"Error processing {yaml_file}: {e}")
            continue

    raise ValueError(f"Prompt '{prompt_name}' not found in {directory_path}")