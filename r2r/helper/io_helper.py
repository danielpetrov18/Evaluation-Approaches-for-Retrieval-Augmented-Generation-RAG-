import os
import json
import yaml
from pathlib import Path
    
def iterate_over_files(folder_path):
    """
    Iterate over all files in a given folder and its subfolders.

    Args:
        folder_path (str): Path to the folder to iterate over.

    Yields:
        str: Path to each file.

    Returns:
        list[str]: List of all file names in the folder and its subfolders.
    """
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            yield file_path
            filenames.append(file)
    return filenames
    
def parse_r2r_error(r2r_exc):
    """
    Parse the R2R exception message.

    Args:
        r2r_exc (R2RException): The R2R exception object containing the error message.

    Returns:
        str: The parsed error message.
    """
    try:
        parsed_err_msg = json.loads(r2r_exc.message)
        return parsed_err_msg['detail']['message']
    except (json.JSONDecodeError, KeyError) as e:
        raise Exception(e)

def load_prompt(directory_path, prompt_name):
    if not directory_path:
        raise ValueError("No directory path provided")
    
    if not directory_path.is_dir():
        error_msg = f"The specified path is not a directory: {directory_path}"
        raise ValueError(error_msg)

    for yaml_file in directory_path.glob("*.yaml"):
        try:
            with open(yaml_file, "r") as file:
                data = yaml.safe_load(file)
                if not data:
                    print(f"Warning: YAML file {yaml_file} is empty")
                    continue
                if not isinstance(data, dict):
                    raise ValueError(f"Invalid format in YAML file {yaml_file}")
                
                for name, prompt_data in data.items():
                    if name == prompt_name:
                          return prompt_data
 
        except yaml.YAMLError as e:
            error_msg = f"Error loading prompts from YAML file {yaml_file}: {e}"
            raise ValueError(error_msg)
        except KeyError as e:
            error_msg = f"Missing key in YAML file {yaml_file}: {e}"
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error loading YAML file {yaml_file}: {e}"
            raise ValueError(error_msg)