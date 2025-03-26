"""Enables the user to manage prompts with GUI."""

# pylint: disable=E0401
# pylint: disable=C0301

import streamlit as st
from st_app import load_client
from backend.prompts import (
    list_prompts,
    create_prompt,
    delete_prompt
)

if __name__ == "__page__":
    st.title("📝 Prompt Management")

    tab_list, tab_create, tab_delete = st.tabs(
        [
            "List Prompts", 
            "Create Prompt", 
            "Delete Prompt"
        ]
    )

    with tab_list:
        st.markdown("**List Prompts**")

        if st.button(label="Fetch Prompt List", key="fetch_prompts_btn"):
            list_prompts(load_client())

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
        if st.button(label="Create Prompt"):
            if uploaded_file is None:
                st.error(body="Please upload a YAML file.")
            else:
                create_prompt(load_client(), uploaded_file)

    with tab_delete:
        st.markdown("**Delete Prompt by Name**")

        del_prompt_name = st.text_input(
            label="Prompt Name to Delete",
            placeholder="Ex. prompt_name",
            value=""
        )
        if st.button("Delete Prompt", key="delete_prompt_btn"):
            if not del_prompt_name.strip():
                st.error("Please enter a prompt name.")
            else:
                delete_prompt(load_client(), del_prompt_name.strip())
