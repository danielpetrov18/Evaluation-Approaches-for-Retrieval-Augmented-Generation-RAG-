"""GUI support for managing indices."""

from pathlib import Path
from dataclasses import dataclass
import yaml
import streamlit as st
from streamlit.errors import Error
from r2r import R2RException
from st_app import load_client # pylint: disable=E0401

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

def list_indices():
    """Fetch all available indices"""
    try:
        indices = load_client().indices.list().results.indices
        if indices:
            for obj in indices:
                with st.expander(
                    label=f"Index: {obj.index['name']}",
                    expanded=False
                ):
                    st.json(obj)
        else:
            st.info("No indices found.")
    except R2RException as r2re:
        st.error(f"Error listing indices: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def _construct_index_config(
    index_method: str,
    index_name: str,
    index_measure: str,
    index_arguments: dict
) -> dict:
    """Helper function to construct index configuration."""
    # https://medium.com/@emreks/comparing-ivfflat-and-hnsw-with-pgvector-performance-analysis-on-diverse-datasets-e1626505bc9a
    # https://towardsdatascience.com/similarity-search-part-4-hierarchical-navigable-small-world-hnsw-2aad4fe87d37/

    if index_method not in ('hnsw', 'ivf_flat'):
        raise ValueError('[-] Invalid index method, only hnsw and ivf_flat are supported! [-]')

    if index_measure not in ('ip_distance', 'l2_distance', 'cosine_distance'):
        raise ValueError('[-] Only ip_distance, l2_distance and cosine_distance are supported!')

    config = {
        # According to the documentation it should be vectors. However, it doesn't work.
        'table_name': 'chunks',
        'index_method': index_method,
        'index_measure': index_measure,
        'index_arguments': index_arguments,
        'index_name': index_name,
        # According documentaition it should be 'embedding', however it doesn't work.
        # I've established connection to the pgvector container. It should be 'vec'.
        'index_column': 'vec', 
        'concurrently': True
    }
    return config

def create_idx(file):
    """Create an index from an uploaded YAML file."""

    target_path = Path(st.session_state['indices_dir']) / file.name

    # Check if file already exists
    if target_path.exists():
        st.error(
            "A file with this name already exists. Please rename your file and try again.", 
            icon="⚠️"
        )
    else:
        # Save the uploaded file (uploaded as bytes)
        with open(file=target_path, mode='wb') as f:
            f.write(file.getbuffer())

        try:
            # Parse the config
            index = load_index_config_from_yaml(str(target_path))

            idx_config = _construct_index_config(
                index_method=index.method,
                index_name=index.name,
                index_measure=index.measure,
                index_arguments=index.arguments
            )

            idx_creation_resp = load_client().indices.create(
                config=idx_config,
                run_with_orchestration=True
            )
            st.success(idx_creation_resp.results.message, icon="✅")
        except ValueError as ve:
            target_path.unlink(missing_ok=True)
            st.error(f"Error in YAML file structure: {str(ve)}")
        except R2RException as r2re:
            target_path.unlink(missing_ok=True)
            st.error(f"Error creating index: {str(r2re)}")
        except Error as e:
            target_path.unlink(missing_ok=True)
            st.error(f"Unexpected error: {str(e)}")

def retrieve_idx(retrieve_name: str):
    """Retrieves the index if it exists."""
    try:
        index_data = load_client().indices.retrieve(retrieve_name, table_name="chunks").results
        if index_data.index['name'] == retrieve_name:
            st.markdown("**Index Details:**")
            st.json(index_data)
        else:
            st.info("Index not found.")
    except R2RException as r2re:
        st.error(f"Error retrieving index: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def delete_idx(name: str):
    """Delete index if available."""
    try:
        result = load_client().indices.delete(
            index_name=name,
            table_name="chunks"
        ).results.message

        index_dir = Path(st.session_state['indices_dir'])
        for file in index_dir.iterdir():
            fullpath = str(index_dir / file.name)
            idx_obj = load_index_config_from_yaml(fullpath)
            if idx_obj.name == name.strip():
                file.unlink(missing_ok=True)
                break
        st.success(body=result, icon="🗑️")
    except R2RException as r2re:
        st.error(f"Error deleting index: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

if __name__ == "__page__":
    st.title("📊 Indices")

    tab_list, tab_create, tab_retrieve, tab_delete = st.tabs(
        [
            "List Indices",
            "Create Index",
            "Retrieve Index",
            "Delete Index"
        ]
    )

    with tab_list:
        st.markdown("**List Indices**")

        if st.button(label="Fetch Indices"):
            list_indices()

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

        if create_btn:
            if not uploaded_file:
                st.error("Please upload a YAML file to create an index.", icon="⚠️")
            else:
                create_idx(uploaded_file)

    with tab_retrieve:
        st.markdown("**Retrieve Index by Name**")

        chosen_idx = st.text_input(
            label="Index Name to Retrieve",
            placeholder="Ex. index_name",
            value=""
        )

        if st.button(label="Get Index Details"):
            if not chosen_idx.strip():
                st.error("Please provide an index name to retrieve.", icon="⚠️")
            else:
                retrieve_idx(chosen_idx.strip())

    with tab_delete:
        st.markdown("**Delete Index by Name**")

        del_index_name = st.text_input(
            label="Index Name to Delete",
            value="",
            placeholder="Ex index_name"
        )
        if st.button(label="Delete Index"):
            if not del_index_name.strip():
                st.error("Please provide an index name to delete.", icon="⚠️")
            else:
                delete_idx(del_index_name.strip())
