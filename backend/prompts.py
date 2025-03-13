"""
This modules helps the user interact with prompts that are part of the database.
"""

import logging
import dataclasses
import yaml
from r2r import R2RClient, R2RException

@dataclasses.dataclass
class MyPrompt:
    """Custom class to encapsulate prompt information."""
    name: str
    template: str
    input_types: dict

class Prompts:
    """
    This class supports functionality for adding, listing, deleting and updating prompts.
    """

    def __init__(self, client: R2RClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

    def list_prompts(self):
        """
        Retrieve a list of prompts from the R2R service. 

        Returns:
            WrappedPromptsResponse: List of prompts in the R2R service.

        Raises:
            R2RException: If there is an error while fetching the list of prompts.
            Exception: If an unexpected error occurs.
        """
        try:
            prompts_resp = self._client.prompts.list()
            return prompts_resp
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while listing prompts: %s [-]', str(e))
            raise

    def create_prompt(self, filepath: str):
        """
        Create a prompt in the R2R service. 
        The prompt is identified by its name, and has a template and input types.
        
        Args:
            filepath (str): Path to the YAML file containing the prompt data.
        
        Returns:
            WrappedGenericMessageResponse: Response from the R2R service.
        
        Raises:
            R2RException: If there is an error while creating the prompt.
            ValueError: If the YAML file is invalid.
            Exception: If an unexpected error occurs.
        """
        try:
            prompt = self._load_prompt_from_yaml(filepath)
            prompt_resp = self._client.prompts.create(
                name=prompt.name,
                template=prompt.template,
                input_types=prompt.input_types
            )
            return prompt_resp
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except ValueError as ve:
            self._logger.error(str(ve))
            raise ValueError(str(ve)) from ve
        except Exception as e:
            self._logger.error('[-] Unexpected error while creating prompt: %s [-]', str(e))
            raise

    def _load_prompt_from_yaml(self, filepath: str):
        """
        Loads a prompt from a YAML file.

        The YAML file should contain a single top-level key that represents the prompt name.
        The value should be a dictionary containing a 'template' key for the prompt template,
        and an 'input_types' key that is a dictionary mapping input names to input types.

        Args:
            filepath (str): The path to the YAML file containing the prompt data.

        Returns:
            MyPrompt: Instance containing the name, template, and input types.

        Raises:
            ValueError: If the YAML file is invalid.
            Exception: If an unexpected error occurs.
        """
        try:
            with open(file=filepath, mode='r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # There should be exactly one top-level key that represents the prompt name.
            if not isinstance(data, dict) or len(data.keys()) != 1:
                raise ValueError(
                    "YAML file must contain exactly one top-level key representing the prompt name!"
                )

            name = list(data.keys())[0]
            prompt_data = data[name] # Template and input types

            if 'template' not in prompt_data or 'input_types' not in prompt_data:
                raise ValueError("The top-level key must contain 'template' and 'input_types'!")

            template = prompt_data['template']
            input_types = prompt_data['input_types']

            if not name or not template or not input_types:
                raise ValueError("YAML file must contain 'name', 'template', and 'input_types'.")

            return MyPrompt(name, template, input_types)
        except Exception as e:
            self._logger.error('[-] Error while loading YAML file: %s [-]', e)
            raise

    def get_prompt_by_name(self, name: str):
        """
        Retrieve a prompt from the R2R service by its name.
        
        Args:
            name (str): Name of the prompt.
        
        Returns:
            WrappedPromptResponse: Response from the R2R service.
        
        Raises:
            R2RException: If there is an error while getting the prompt.
            Exception: If an unexpected error occurs.
        """
        try:
            prompt_resp = self._client.prompts.retrieve(name)
            return prompt_resp
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while getting prompt: %s [-]', str(e))
            raise

    def delete_prompt_by_name(self, name: str):
        """
        Delete a prompt from the R2R service by its name.
        
        Args:
            name (str): Name of the prompt.
        
        Returns:
            WrappedBooleanResponse: Response from the R2R service.
        
        Raises:
            R2RException: If there is an error while deleting the prompt.
            Exception: If an unexpected error occurs.
        """
        try:
            deletion_resp = self._client.prompts.delete(name)
            return deletion_resp
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while deleting prompt: %s [-]', str(e))
            raise
