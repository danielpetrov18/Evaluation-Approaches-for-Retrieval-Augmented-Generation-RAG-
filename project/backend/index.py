# pylint: disable=C0114
# pylint: disable=W0718
# pylint: disable=R1732

import tempfile
import dataclasses
from pathlib import Path
from typing import Dict, Union

import yaml
from r2r import R2RException, R2RClient

import streamlit as st
from streamlit.errors import Error
from streamlit.runtime.uploaded_file_manager import UploadedFile

@dataclasses.dataclass
class Index:
    """Following a more object oriented approach by grouping data together."""
    name: str
    method: str
    measure: str
    arguments: Dict[str, Union[str, Dict]]

def list_indices(client: R2RClient):
    """Fetch all available indices"""
    try:
        indices = client.indices.list().results.indices
        if indices:
            for obj in indices:
                with st.expander(
                    label=f"Index: {obj.index['name']}",
                    expanded=False
                ):
                    st.json(obj)
            st.info("You've reached the end of the indices.")
        else:
            st.info("No indices found.")
    except R2RException as r2re:
        st.error(f"Error listing indices: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def create_idx(client: R2RClient, file: UploadedFile):
    """Create an index from an uploaded YAML file."""

    # Save uploaded file temporarily so as to read the data into dict
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".yaml")
    temp_file.write(file.getbuffer())
    temp_file.flush()

    try:
        index = _load_index_config_from_yaml(temp_file.name)

        idx_config = _construct_index_config(
            index_method=index.method,
            index_name=index.name,
            index_measure=index.measure,
            index_arguments=index.arguments
        )

        idx_creation_resp = client.indices.create(
            config=idx_config,
            run_with_orchestration=True
        ).results.message
        st.success(idx_creation_resp)
    except ValueError as ve:
        st.error(f"Error in YAML file structure: {str(ve)}")
    except R2RException as r2re:
        st.error(f"Error creating index: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")
    finally:
        temp_file.close()

def delete_idx(client: R2RClient, name: str):
    """Delete index if available."""
    try:
        result = client.indices.delete(
            index_name=name,
            table_name="chunks"
        ).results.message

        st.success(body=result)
    except R2RException as r2re:
        st.error(f"Error deleting index: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def _load_index_config_from_yaml(filepath: Union[str,Path]) -> Index:
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
    with open(file=filepath, mode='r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, Dict) or len(data.keys()) != 1:
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

def _construct_index_config(
    index_method: str,
    index_name: str,
    index_measure: str,
    index_arguments: dict
) -> Dict[str, Union[str, bool]]:
    """Helper function to construct index configuration."""
    # https://medium.com/@emreks/comparing-ivfflat-and-hnsw-with-pgvector-performance-analysis-on-diverse-datasets-e1626505bc9a
    # https://towardsdatascience.com/similarity-search-part-4-hierarchical-navigable-small-world-hnsw-2aad4fe87d37/

    if index_method not in ('hnsw', 'ivf_flat'):
        raise ValueError('Invalid index method, only hnsw and ivf_flat are supported!')

    if index_measure not in ('ip_distance', 'l2_distance', 'cosine_distance'):
        raise ValueError('Only ip_distance, l2_distance and cosine_distance are supported!')

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
