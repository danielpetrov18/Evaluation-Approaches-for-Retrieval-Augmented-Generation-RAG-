"""Enables the user to manage prompts with GUI."""

import sys
from pathlib import Path
import yaml
import streamlit as st
from r2r import R2RException
from st_app import load_client
from streamlit.errors import Error

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from prompts import Prompts

@st.cache_resource
def get_prompts_handler():
    """Get prompts handler."""
    return Prompts(client=load_client())

if __name__ == "__page__":
    st.title("📝 Prompt Management")

    tab_list, tab_create, tab_retrieve, tab_delete = st.tabs(
        [
            "List Prompts", 
            "Create Prompt", 
            "Retrieve Prompt", 
            "Delete Prompt"
        ]
    )

    with tab_list:
        st.markdown("**List Prompts**")
        list_btn = st.button("Fetch Prompt List")
        if list_btn:
            try:
                prompts = get_prompts_handler().list_prompts().results
                if prompts:
                    st.write(f"Found {len(prompts)} prompts:")
                    for prompt in prompts:
                        st.json(prompt)
                else:
                    st.info("No prompts found.")
            except R2RException as r2re:
                st.error(f"Error listing prompts: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

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
        create_btn = st.button("Create Prompt")

        if create_btn and uploaded_file is not None:
            template_dir = Path(backend_dir) / "prompts"
            template_dir.mkdir(parents=True, exist_ok=True)

            target_path = template_dir / uploaded_file.name
            if target_path.exists():
                st.error(
                    body="A file with this name exists. Please rename your file and try again.",
                    icon="⚠️"
                )
            else:
                with open(file=target_path, mode='wb') as f:
                    f.write(uploaded_file.getbuffer())

                with open(file=target_path, mode='r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                index_name = list(data.keys())[0] # The key is top-level component

                try:
                    # Try to see if such a prompt exists.
                    # If it actually exists remove the uploaded file.
                    # Otherwise an exception will be thrown, caught and the prompt
                    #    will be added to the backend.
                    name = get_prompts_handler().get_prompt_by_name(index_name).results.name
                    target_path.unlink(missing_ok=True)
                    st.error(
                        body=f"A prompt with name '{index_name}' already exists.",
                        icon="⚠️"
                    )
                except R2RException as outer_r2re:
                    try:
                        result = get_prompts_handler().create_prompt(str(target_path))
                        st.success(
                            body=f"Prompt created successfully! Message: {result.results.message}",
                            icon="✅"
                        )
                    except R2RException as r2re:
                        st.error(f"Error creating prompt: {str(r2re)}")
                    except Error as e:
                        st.error(f"Unexpected error: {str(e)}")

    with tab_retrieve:
        st.markdown("**Retrieve Prompt by Name**")
        prompt_name = st.text_input(label="Prompt Name to Retrieve")
        retrieve_btn = st.button("Get Prompt")

        if retrieve_btn and prompt_name.strip():
            try:
                prompt_data = get_prompts_handler().get_prompt_by_name(prompt_name.strip()).results
                if prompt_data:
                    st.write("Prompt Data:")
                    st.json(prompt_data)
                else:
                    st.info("Prompt not found.")
            except R2RException as r2re:
                st.error(f"Error retrieving prompt: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

    with tab_delete:
        st.markdown("**Delete Prompt by Name**")
        del_prompt_name = st.text_input(
            label="Prompt Name to Delete",
            placeholder="Ex. example_prompt_name"
        )
        del_btn = st.button("Delete Prompt")

        if del_btn and del_prompt_name.strip():
            try:
                result = get_prompts_handler().delete_prompt_by_name(del_prompt_name.strip())

                prompt_dir = Path(backend_dir) / "prompts"
                for file in prompt_dir.iterdir():
                    FULLPATH = str(prompt_dir / file.name)
                    with open(file=FULLPATH, mode='r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)

                    prompt_name = list(data.keys())[0]
                    if prompt_name == del_prompt_name.strip():
                        file.unlink(missing_ok=True)
                        break

                st.success(f"Prompt deletion result: {result.results}", icon="🗑️")
            except R2RException as r2re:
                st.error(f"Error deleting prompt: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")
