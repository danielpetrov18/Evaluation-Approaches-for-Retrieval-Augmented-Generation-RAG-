"""GUI support for managing indices."""

import sys
from pathlib import Path
from dataclasses import dataclass
import yaml
import streamlit as st
from streamlit.errors import Error
from r2r import R2RException
from st_app import load_client

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from indices import Indices

@st.cache_resource
def get_index_handler():
    """Get the index handler."""
    return Indices(client=load_client())

@dataclass
class Index:
    """Following a more object oriented approach by encapsulating."""
    name: str
    method: str
    measure: str
    arguments: dict

def load_index_config_from_yaml(filepath: str | Path) -> Index:
    """
    Load index configuration from a YAML file.

    Expected structure:
    index_name:
    index_method: <hnsw|ivf_flat>
    index_measure: <ip_distance|l2_distance|cosine_distance>
    index_arguments: dict of arguments or empty

    Returns:
        Index object
    """
    with open(file=filepath, mode='r', encoding='utf-8') as file_handle:
        data = yaml.safe_load(file_handle)

    if not isinstance(data, dict) or len(data.keys()) != 1:
        raise ValueError(
            "YAML file must contain exactly one top-level key representing the index name!"
        )

    idx_name = list(data.keys())[0]
    config_data = data[idx_name]

    if 'index_method' not in config_data or 'index_measure' not in config_data:
        raise ValueError(
            "The top-level key must contain 'index_method' and 'index_measure' fields."
        )

    idx_method = config_data['index_method']
    idx_measure = config_data['index_measure']
    idx_args = config_data.get('index_arguments', {})

    if not idx_name or not idx_method or not idx_measure:
        raise ValueError("YAML file must contain index_name, index_method, and index_measure.")

    return Index(idx_name, idx_method, idx_measure, idx_args)

if __name__ == "__page__":
    st.title("📊 Index Management")

    tab_list, tab_create, tab_retrieve, tab_delete = st.tabs(
        ["List Indices", "Create Index", "Retrieve Index", "Delete Index"]
    )

    with tab_list:
        st.markdown("**List Indices**")
        list_btn = st.button("Fetch Indices")
        if list_btn:
            try:
                indices = get_index_handler().list_indices().results.indices
                if indices:
                    st.write(f"Found {len(indices)} indices:")
                    for index in indices:
                        st.json(index)
                else:
                    st.info("No indices found.")
            except R2RException as r2re:
                st.error(f"Error listing indices: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

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
        create_btn = st.button("Create Index")

        if create_btn and uploaded_file is not None:
            index_config_dir = Path(backend_dir) / "indices"

            # Construct the target file path
            target_path = index_config_dir / uploaded_file.name

            # Check if file already exists
            if target_path.exists():
                st.error(
                    "A file with this name already exists. Please rename your file and try again.", 
                    icon="⚠️"
                )
            else:
                # Save the uploaded file (uploaded as bytes)
                with open(target_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    # Parse the config
                    index = load_index_config_from_yaml(str(target_path))

                    # Create the index
                    idx_creation_resp = get_index_handler().create_index(
                        index_method=index.method,
                        index_name=index.name,
                        index_measure=index.measure,
                        index_arguments=index.arguments
                    )
                    st.success(
                        f"Index created successfully! Message: {idx_creation_resp.results}",
                        icon="✅"
                    )
                except ValueError as ve:
                    target_path.unlink(missing_ok=True)
                    st.error(f"Error in YAML file structure: {str(ve)}")
                except R2RException as r2re:
                    target_path.unlink(missing_ok=True)
                    st.error(f"Error creating index: {str(r2re)}")
                except Error as e:
                    target_path.unlink(missing_ok=True)
                    st.error(f"Unexpected error: {str(e)}")

    with tab_retrieve:
        st.markdown("**Retrieve Index by Name**")
        chosen_idx = st.text_input("Index Name to Retrieve")
        retrieve_btn = st.button("Get Index Details")

        if retrieve_btn and chosen_idx.strip():
            try:
                index_data = get_index_handler().get_index_details(chosen_idx.strip()).results

                if index_data:
                    st.markdown("**Index Details:**")
                    st.json(index_data)
                else:
                    st.info("Index not found.")
            except R2RException as r2re:
                st.error(f"Error retrieving index: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

    with tab_delete:
        st.markdown("**Delete Index by Name**")
        del_index_name = st.text_input("Index Name to Delete")
        del_btn = st.button("Delete Index")

        if del_btn and del_index_name.strip():
            try:
                result = get_index_handler().delete_index_by_name(del_index_name.strip())

                index_dir = Path(backend_dir) / "indices"
                for file in index_dir.iterdir():
                    FULLPATH = str(index_dir / file.name)
                    idx_obj = load_index_config_from_yaml(FULLPATH)
                    if idx_obj.name == del_index_name.strip():
                        file.unlink(missing_ok=True)
                        break
                st.success(f"Index deletion result: {result}", icon="🗑️")
            except R2RException as r2re:
                st.error(f"Error deleting index: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")
