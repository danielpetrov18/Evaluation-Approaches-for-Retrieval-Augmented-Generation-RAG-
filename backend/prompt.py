import yaml
import logging
from typing import List
from r2r import R2RAsyncClient, R2RException

class PromptHandler:
    
    def __init__(self, client: R2RAsyncClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
    
    async def list_prompts(self) -> List[dict]: 
        """
        Retrieve a list of prompts from the R2R service. Each one contains data like input types, template, name, 
        creation date, etc.

        Returns:
            list[dict]: List of prompts in the R2R service.

        Raises:
            R2RException: If there is an error while fetching the list of prompts.
            Exception: If an unexpected error occurs.
        """
        try:
            prompts = await self._client.prompts.list()
            return prompts['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while listing prompts: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while listing prompts: {e} [-]')
            raise Exception(str(e))
        
    async def create_prompt(self, filepath: str) -> dict:    
        """
        Create a prompt in the R2R service. The prompt is identified by its name, and has a template and input types.
        
        Args:
            filepath (str): Path to the YAML file containing the prompt data.
        
        Returns:
            dict: Response from the R2R service.
        
        Raises:
            R2RException: If there is an error while creating the prompt.
            ValueError: If the YAML file is invalid.
            Exception: If an unexpected error occurs.
        """
        try:
            name, template, input_types = self._load_prompt_from_yaml(filepath)
            prompt = await self._client.prompts.create(
                name=name, 
                template=template, 
                input_types=input_types
            )
            return prompt['results']['message']
        except R2RException as r2re:
            err_msg = f'[-] Error while creating prompt: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except ValueError as ve:
            err_msg = f'[-] Error while creating prompt: {ve} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from ve
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while creating prompt: {e} [-]')
            raise Exception(str(e)) from e
        
    def _load_prompt_from_yaml(self, filepath: str) -> tuple[str, str, dict]:
        """
        Loads a prompt from a YAML file.

        The YAML file should contain a single top-level key that represents the prompt name.
        The value of this key should be a dictionary containing a 'template' key for the prompt template,
        and an 'input_types' key that is a dictionary mapping input names to input types.

        Args:
            filepath (str): The path to the YAML file containing the prompt data.

        Returns:
            tuple[str, str, dict]: A tuple containing the prompt name, template, and input types.

        Raises:
            ValueError: If the YAML file is invalid.
            Exception: If an unexpected error occurs.
        """
        try:
            with open(file=filepath, mode='r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # There should be exactly one top-level key that represents the prompt name.
            if not isinstance(data, dict) or len(data.keys()) != 1:
                raise ValueError("YAML file must contain exactly one top-level key representing the prompt name!")
            
            name = list(data.keys())[0]
            prompt_data = data[name] # Template and input types
            
            if 'template' not in prompt_data or 'input_types' not in prompt_data:
                raise ValueError("The top-level key must contain 'template' and 'input_types' fields.")

            template = prompt_data['template']
            input_types = prompt_data['input_types']

            if not name or not template or not input_types:
                raise ValueError("YAML file must contain 'name', 'template', and 'input_types' keys.")

            return name, template, input_types
        except Exception as e:
            self._logger.error(f'[-] Error while loading YAML file: {e} [-]')
            raise Exception(str(e)) from e
        
    async def get_prompt_by_name(self, name: str) -> dict: 
        """
        Retrieve a prompt from the R2R service by its name.
        
        Args:
            name (str): Name of the prompt.
        
        Returns:
            dict: Response from the R2R service.
        
        Raises:
            R2RException: If there is an error while getting the prompt.
            Exception: If an unexpected error occurs.
        """
        try:
            prompt = await self._client.prompts.retrieve(name)
            return prompt['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while getting prompt: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 404) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while getting prompt: {e} [-]')
            raise Exception(str(e)) from e
    
    async def delete_prompt_by_name(self, name: str) -> dict:   
        """
        Delete a prompt from the R2R service by its name. Should return {success: true}
        
        Args:
            name (str): Name of the prompt.
        
        Returns:
            dict: Response from the R2R service.
        
        Raises:
            R2RException: If there is an error while deleting the prompt.
            Exception: If an unexpected error occurs.
        """
        try:
            prompt = await self._client.prompts.delete(name)
            return prompt['results']['message']
        except R2RException as r2re:
            err_msg = f'[-] Error while deleting prompt: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 404) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while deleting prompt: {e} [-]')
            raise Exception(str(e)) from e