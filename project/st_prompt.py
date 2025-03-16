"""Enables the user to manage prompts with GUI."""

import json
import dataclasses
from pathlib import Path
import yaml
import requests
import streamlit as st
from r2r import R2RException
from st_app import load_client # pylint: disable=E0401
from streamlit.errors import Error

# Disables pylint for long lines (above 100)
# pylint: disable=C0301

# Disable pylint for "too broad Exception"
# pylint: disable=W0703

@dataclasses.dataclass
class MyPrompt:
    """Custom class to encapsulate prompt information."""
    name: str
    template: str
    input_types: dict

def list_prompts():
    """List all available prompts"""
    try:
        prompts = load_client().prompts.list().results
        if prompts:
            st.write(f"Found {len(prompts)} prompts:")
            for prompt in prompts:
                with st.expander(label=f"Prompt: {prompt.name}", expanded=False):
                    st.json(prompt)
        else:
            st.info("No prompts found.")
    except R2RException as r2re:
        st.error(f"Error listing prompts: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def load_prompt_from_yaml(filepath: str):
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

def create_prompt(file):
    """Create a custom propmt and save into database"""

    prompts_dir = st.session_state['prompts_dir']
    prompts_dir = Path(prompts_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    target_path = prompts_dir / file.name
    if target_path.exists():
        st.error(
            body="A file with this name exists. Please rename your file and try again.",
            icon="⚠️"
        )
    else:
        with open(file=target_path, mode='wb') as f:
            f.write(file.getbuffer())

        with open(file=target_path, mode='r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        prompt_name = list(data.keys())[0] # The key is top-level component

        try:
            # Try to see if such a prompt exists.
            # If it actually exists remove the uploaded file.
            # Otherwise an exception will be thrown, caught and the prompt
            #    will be added to the backend.
            name = load_client().prompts.retrieve(prompt_name).results.name
            target_path.unlink(missing_ok=True)
            st.error(
                body=f"A prompt with name '{name}' already exists.",
                icon="⚠️"
            )
        except R2RException:
            try:
                prompt_obj: MyPrompt = load_prompt_from_yaml(target_path)

                if prompt_obj is None:
                    st.error("Error loading prompt from YAML file.")
                else:
                    result = load_client().prompts.create(
                        name = prompt_obj.name,
                        template = prompt_obj.template,
                        input_types = prompt_obj.input_types
                    )
                    st.success(
                        body=f"Prompt created successfully! Message: {result.results.message}",
                        icon="✅"
                    )
            except ValueError as ve:
                st.error(f"Error creating prompt: {str(ve)}")
            except R2RException as r2re:
                st.error(f"Error creating prompt: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

def retrieve(name: str):
    """Get specific prompt by name"""
    try:
        prompt = load_client().prompts.retrieve(name).results
        if prompt:
            st.json(prompt)
        else:
            st.info("Prompt not found.")
    except R2RException as r2re:
        st.error(f"Error retrieving prompt: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def delete(name: str):
    """Delete specific prompt by name"""
    try:
        result = load_client().prompts.delete(name)
        prompt_dir = Path(st.session_state['prompts_dir'])

        # Try to delete file if it exists
        for file in prompt_dir.iterdir():
            fullpath = str(prompt_dir / file.name)
            with open(file=fullpath, mode='r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            prompt_name = list(data.keys())[0]
            if prompt_name == name.strip():
                file.unlink(missing_ok=True)
                break
        st.success(f"Prompt deletion result: {result.results}", icon="🗑️")
    except R2RException as r2re:
        st.error(f"Error deleting prompt: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def update_prompt(name: str, new_template: str, new_input_types: dict):
    """Makes a PUT request to update an existing prompt"""
    try:
        url = f"http://127.0.0.1:7272/v3/prompts/{name}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state['token']}"
        }
        payload = {
            "template": new_template,
            "input_types": new_input_types
        }

        resp = requests.put(
            url = url,
            headers = headers,
            json = payload,
            timeout=5
        )

        if resp.status_code != 200:
            raise R2RException(resp.text, resp.status_code)

        # Now update the YAML file if it exists
        prompts_dir = Path(st.session_state['prompts_dir'])
        file_updated = False

        for file in prompts_dir.iterdir():
            if file.is_file() and file.suffix.lower() in ['.yaml', '.yml']:
                try:
                    with open(file=file, mode='r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)

                    if isinstance(data, dict) and name in data:
                        data[name]['template'] = new_template
                        data[name]['input_types'] = new_input_types

                        with open(file=file, mode='w', encoding='utf-8') as f:
                            yaml.dump(data, f, default_flow_style=False)

                        file_updated = True
                        break
                except Exception as e:
                    st.warning(f"Could not update file {file.name}: {str(e)}")

        if file_updated:
            st.success(
                body=f"Prompt updated successfully in backend and YAML file! {json.loads(resp.text)['results']['message']}",
                icon="✅"
            )
        else:
            st.success(
                body=f"Prompt updated successfully in backend, but no matching YAML file found. {json.loads(resp.text)['results']['message']}",
                icon="✅"
            )

    except R2RException as r2re:
        st.error(f"Error updating prompt: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

if __name__ == "__page__":
    st.title("📝 Prompt Management")

    tab_list, tab_create, tab_retrieve, tab_update, tab_delete = st.tabs(
        [
            "List Prompts", 
            "Create Prompt", 
            "Retrieve Prompt", 
            "Update Prompt",
            "Delete Prompt"
        ]
    )

    with tab_list:
        st.markdown("**List Prompts**")

        if st.button(label="Fetch Prompt List"):
            list_prompts()

    with tab_create:
        st.markdown("**Create Prompt from YAML**")

        with st.expander("Upload Instructions", expanded=False):
            st.markdown("""
            ### YAML File Requirements
            
            Upload a YAML file that defines a prompt:
            """)

            st.code("""
                # Example prompt template
                example_prompt_name: # Required
                    template: >      # Required
                        Hello, {name}!
                    
                    input_types:     # Required
                        name: string
                    overwrite_on_diff: true
            """, language="yaml", line_numbers=True)

        uploaded_file = st.file_uploader(
            label="Upload YAML Prompt File",
            type=["yaml", "yml"]
        )
        if st.button(label="Create Prompt") and uploaded_file is not None:
            if uploaded_file is None:
                st.error(
                    body="Please upload a YAML file.",
                    icon="⚠️"
                )
            else:
                create_prompt(uploaded_file)

    with tab_retrieve:
        st.markdown("**Retrieve Prompt by Name**")

        p_name = st.text_input(
            label="Prompt Name to Retrieve",
            placeholder="Ex. prompt_name",
            value=""
        )
        if st.button(label="Get Prompt"):
            if not p_name.strip():
                st.error("Please enter a prompt name.")
            else:
                retrieve(p_name.strip())

    with tab_update:
        st.markdown("**Update Existing Prompt**")

        update_prompt_name = st.text_input(
            label="Prompt Name to Update",
            placeholder="Ex. prompt_name",
            value=""
        )

        if not update_prompt_name.strip():
            st.error("Please enter a prompt name.")
        else:
            try:
                existing_prompt = load_client().prompts.retrieve(update_prompt_name.strip()).results
                if existing_prompt:
                    prompt_dict = existing_prompt.model_dump()

                    # Extract template and input_types
                    current_template = prompt_dict.get('template', '')
                    current_input_types = prompt_dict.get('input_types', {})

                    # Show the template for editing
                    updated_template = st.text_area(
                        label="Updated Template",
                        value=current_template,
                        height=200
                    )

                    # Show the input_types for editing
                    st.markdown("**Input Types (JSON format)**")
                    input_types_str = st.text_area(
                        label="Input Types",
                        value=str(current_input_types).replace("'", "\""),
                        height=150
                    )

                    if st.button("Confirm"):
                        try:
                            update_prompt(
                                update_prompt_name.strip(),
                                updated_template,
                                json.loads(input_types_str)
                            )
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format for Input Types. Please check your syntax.")
                else:
                    st.info(f"Prompt '{update_prompt_name}' not found.")
            except R2RException as r2re:
                st.error(f"Error retrieving prompt: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

    with tab_delete:
        st.markdown("**Delete Prompt by Name**")

        del_prompt_name = st.text_input(
            label="Prompt Name to Delete",
            placeholder="Ex. prompt_name",
            value=""
        )
        if st.button("Delete Prompt"):
            if not del_prompt_name.strip():
                st.error("Please enter a prompt name.")
            else:
                delete(del_prompt_name.strip())
