"""
GUI support for managing indices.
Indices can make semantic search much faster and efficient.
Both HNSW and IVF_FLAT are supported.

https://r2r-docs.sciphi.ai/api-and-sdks/indices/indices
"""

# pylint: disable=E0401

import streamlit as st
from backend.index import (
    list_indices,
    create_idx,
    delete_idx
)
from st_app import r2r_client

if __name__ == "__page__":
    st.title("📊 Indices")

    tab_list, tab_create, tab_delete = st.tabs(
        [
            "List Indices",
            "Create Index",
            "Delete Index"
        ]
    )

    with tab_list:
        st.markdown("**List Indices**")

        if st.button(label="Fetch Indices", key="fetch_indices_btn"):
            list_indices(r2r_client())

    with tab_create:
        st.markdown("**Create Index from YAML**")

        with st.expander("Upload Instructions", expanded=False):
            st.markdown("""
            ### YAML File Requirements
            
            Upload a YAML file that defines an index configuration with the following structure:
            """)

            # Code block showing the expected format
            st.code("""
                # Example index configuration
                my_index_name: # Required
                    index_method: hnsw  # Required
                    index_measure: cosine_distance # Required
                    index_arguments:
                        m: 16
                        ef_construction: 64
                        ef: 40 
            """, language="yaml", line_numbers=True)

        uploaded_file = st.file_uploader(
            label="Upload YAML Index File",
            type=["yaml", "yml"]
        )

        if st.button(label="Create Index", key="create_index_btn"):
            if not uploaded_file:
                st.error("Please upload a YAML file to create an index.")
            else:
                create_idx(r2r_client(), uploaded_file)

    with tab_delete:
        st.markdown("**Delete Index by Name**")

        del_index_name = st.text_input(
            label="Index Name to Delete",
            placeholder="Ex index_name",
            value=""
        )
        if st.button(label="Delete Index", key="delete_index_btn"):
            if not del_index_name.strip():
                st.error("Please provide an index name to delete.")
            else:
                delete_idx(r2r_client(), del_index_name.strip())
