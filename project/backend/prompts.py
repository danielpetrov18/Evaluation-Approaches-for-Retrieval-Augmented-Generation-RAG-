"""Enables the user to manage prompts with GUI."""

# pylint: disable=R1732
# pylint: disable=W0718

import tempfile
import dataclasses
import yaml
from r2r import (
    R2RException,
    R2RClient
)
import streamlit as st
from streamlit.errors import Error
from streamlit.runtime.uploaded_file_manager import UploadedFile

@dataclasses.dataclass
class MyPrompt:
    """Custom class to group prompt information."""
    name: str
    template: str
    input_types: dict

def list_prompts(client: R2RClient):
    """List all available prompts"""
    try:
        prompts = client.prompts.list().results
        if prompts:
            st.write(f"Found {len(prompts)} prompts:")
            for prompt in prompts:
                with st.expander(label=f"Prompt: {prompt.name}", expanded=False):
                    st.json(prompt)
            st.info("You've reached the end of the prompts.")
        else:
            st.info("No prompts found.")
    except R2RException as r2re:
        st.error(f"Error listing prompts: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def create_prompt(client: R2RClient, file: UploadedFile):
    """Create a custom prompt and save into database"""

    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".yaml")
    temp_file.write(file.getbuffer())
    temp_file.flush()

    try:
        prompt_obj: MyPrompt = _load_prompt_from_yaml(temp_file.name)

        if not prompt_obj:
            st.error("Error loading prompt from YAML file.")
        else:
            if _check_prompt_exists(client, prompt_obj.name):
                st.error(f"Prompt with name {prompt_obj.name} already exists.")
            else:
                result = client.prompts.create(
                    name = prompt_obj.name,
                    template = prompt_obj.template,
                    input_types = prompt_obj.input_types
                ).results.message
                st.success(result)
    except ValueError as ve:
        st.error(f"Error creating prompt: {str(ve)}")
    except R2RException as r2re:
        st.error(f"Error creating prompt: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")
    finally:
        temp_file.close()

def delete_prompt(client:R2RClient, name: str):
    """Delete specific prompt by name"""
    try:
        result = client.prompts.delete(name)
        st.success(f"Prompt deletion result: {result.results}")
    except R2RException as r2re:
        st.error(f"Error deleting prompt: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def _load_prompt_from_yaml(filepath: str) -> MyPrompt | None:
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
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")
        return None

def _check_prompt_exists(client: R2RClient, name: str) -> bool:
    try:
        prompt = client.prompts.retrieve(name).results
        return prompt is not None
    except R2RException:
        return False
