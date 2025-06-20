# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=R1732
# pylint: disable=W0718

import tempfile
import dataclasses
from datetime import datetime
from typing import Union, Dict, List, Any

import yaml
import requests
import streamlit as st
from streamlit.errors import Error
from streamlit.runtime.uploaded_file_manager import UploadedFile

@dataclasses.dataclass
class MyPrompt:
    name: str
    template: str
    input_types: Dict[str, Union[str, Dict]]

def list_prompts():
    response: requests.Response = requests.get(
        url="http://r2r:7272/v3/prompts",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}",
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Failed to retrieve prompts: {response.status_code} - {response.text}")
        return

    prompts: List[Dict] = response.json().get("results", [])

    if not prompts:
        st.info("No prompts found.")
        return

    st.subheader("Available Prompts")

    for prompt in prompts:
        with st.expander(label=f"📝 {prompt['name']}", expanded=False):
            st.markdown(f"**ID**: `{prompt['id']}`")
            input_types_str: str = ', '.join(f'{k}: {v}' for k, v in prompt['input_types'].items())
            st.markdown(f"**Input Types**: `{input_types_str if input_types_str else 'None'}`")

            # Format timestamps
            created = datetime.fromisoformat(prompt['created_at'].replace("Z", "+00:00"))
            updated = datetime.fromisoformat(prompt['updated_at'].replace("Z", "+00:00"))

            st.markdown(f"**Created**: {created.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            st.markdown(f"**Updated**: {updated.strftime('%Y-%m-%d %H:%M:%S')} UTC")

            st.markdown("**Template:**")
            st.code(prompt['template'], language="jinja2", line_numbers=True)

            delete_doc_btn = st.button(
                label="❌ Delete Prompt",
                key=f"delete_{prompt['id']}",
                on_click=delete_prompt,
                args=(prompt['name'], )
            )

    st.info("You've reached the end of the prompts.")

def create_prompt(file: UploadedFile):
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".yaml")
    temp_file.write(file.getbuffer())
    temp_file.flush()

    try:
        prompt_obj: MyPrompt = _load_prompt_from_yaml(temp_file.name)

        if not prompt_obj:
            st.error("Error loading prompt from YAML file.")
            return

        if _check_prompt_exists(prompt_obj.name):
            st.error(f"Prompt with name {prompt_obj.name} already exists!")
            return

        response: requests.Response = requests.post(
            url="http://r2r:7272/v3/prompts",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {st.session_state['bearer_token']}"
            },
            json={
                "name": prompt_obj.name,
                "template": prompt_obj.template,
                "input_types": prompt_obj.input_types
            },
            timeout=5
        )

        if response.status_code != 200:
            st.error(f"Failed to retrieve prompts: {response.status_code} - {response.text}")
            return

        st.success(response.json()['results']['message'])
    except ValueError as ve:
        st.error(f"Error creating prompt: {str(ve)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")
    finally:
        temp_file.close()

def delete_prompt(name: str):
    try:
        response: requests.Response = requests.delete(
            url=f"http://r2r:7272/v3/prompts/{name}",
            headers={
                "Authorization": f"Bearer {st.session_state['bearer_token']}",
            },
            timeout=5
        )

        if response.status_code != 200:
            st.error(f"Failed to delete prompt: {response.status_code} - {response.text}")
            return

        st.success(f"Prompt '{name}' deleted successfully.")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def _load_prompt_from_yaml(filepath: str) -> Union[MyPrompt,None]:
    """
    Loads a prompt from a YAML file.

    The YAML file should contain a single top-level key that represents the prompt name.
    The value should be a dictionary containing a 'template' key for the prompt template,
    and an 'input_types' key that is a dictionary mapping input names to input types.

    Args:
        filepath (str): The path to the YAML file containing the prompt data.

    Returns:
        MyPrompt or None: Instance containing the name, template, and input types.

    Raises:
        ValueError: If the YAML file is invalid.
        Exception: If an unexpected error occurs.
    """
    try:
        with open(file=filepath, mode='r', encoding='utf-8') as f:
            data: Any = yaml.safe_load(f)

        # There should be exactly one top-level key that represents the prompt name.
        if not isinstance(data, Dict) or len(data.keys()) != 1:
            raise ValueError(
                "YAML file must contain exactly one top-level key representing the prompt name!"
            )

        name: str = list(data.keys())[0]
        prompt_data: Any = data[name] # Template and input types

        if 'template' not in prompt_data or 'input_types' not in prompt_data:
            raise ValueError("The top-level key must contain 'template' and 'input_types'!")

        template: str = prompt_data['template']
        input_types: str = prompt_data['input_types']

        if not name or not template or not input_types:
            raise ValueError("YAML file must contain 'name', 'template', and 'input_types'.")

        return MyPrompt(name, template, input_types)
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")
        return None

def _check_prompt_exists(name: str) -> bool:
    response: requests.Response = requests.post(
        url=f"http://r2r:7272/v3/prompts/{name}",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        timeout=5
    )

    if response.status_code == 200:
        return True

    return False
