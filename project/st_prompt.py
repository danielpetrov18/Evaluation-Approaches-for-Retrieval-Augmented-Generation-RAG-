"""
Prompt templating can significantly improve the quality of responses.
It's worth trying out different prompts for RAG to see which one works best.
Prompts can be defined in YAML format. Check out the prompts page in the browser.
"""

# pylint: disable=E0401
# pylint: disable=C0301

from typing import Union

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from backend.prompt import (
    list_prompts,
    create_prompt,
    delete_prompt
)
from st_app import r2r_client

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
            list_prompts(r2r_client())

    with tab_create:
        st.markdown("**Create Prompt from YAML**")

        with st.expander("Upload Instructions", expanded=False):
            st.markdown("""
            ### YAML File Requirements
            
            Upload a YAML file that defines a prompt:
            """)

            st.code("""
                custom_rag:
                    template: |
                        You are a helpful assistant. Use only the information in the context below to answer the user's question.

                        Do not use any other knowledge you may have been trained on.

                        If the context does not have the information needed to answer the question, say that you cannot answer based on the available information.

                        Do not include citations or references to specific lines or parts of the context.

                        Always keep your answer relevant and focused on the user's question.

                        ### Context:
                        {context}

                        ### Query:
                        {query}

                        ## Response:
                    input_types:
                        query: str
                        context: str
            """, language="yaml", line_numbers=True, wrap_lines=True)

        uploaded_file: Union[UploadedFile, None] = st.file_uploader(
            label="Upload YAML Prompt File",
            type=["yaml", "yml"]
        )
        if st.button(label="Create Prompt", key="create_prompt_btn"):
            if uploaded_file is None:
                st.error(body="Please upload a YAML file.")
            else:
                create_prompt(r2r_client(), uploaded_file)

    with tab_delete:
        st.markdown("**Delete Prompt by Name**")

        del_prompt_name: str = st.text_input(
            label="Prompt Name to Delete",
            placeholder="Ex. prompt_name",
            value=""
        )
        if st.button("Delete Prompt", key="delete_prompt_btn"):
            del_prompt_name: str = del_prompt_name.strip()
            if not del_prompt_name:
                st.error("Please enter a prompt name.")
            else:
                delete_prompt(r2r_client(), del_prompt_name)
