# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0303
# pylint: disable=W0718
# pylint: disable=R1732

import tempfile
import dataclasses
from pathlib import Path
from typing import Dict, Union, List, Any

import yaml
import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

@dataclasses.dataclass
class Index:
    name: str
    method: str
    measure: str
    arguments: Dict[str, Union[str, Dict]]

def list_indices():
    response: requests.Response = requests.get(
        url="http://r2r:7272/v3/indices",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Failed to fetch indices: {response.status_code} - {response.text}")
        return

    indices: List[Dict[str, Any]] = response.json()['results'].get('indices', [])

    if not indices:
        st.info("No indices found.")
        return

    for obj in indices:
        index = obj['index']

        with st.expander(label=f"🧠 Index: `{index['name']}`", expanded=False):
            st.markdown(f"""
**Table Name**: `{index['table_name']}`  
**Size**: `{index['size_in_bytes']:,} bytes`  
**Row Estimate**: `{index['row_estimate']}`  
**Scans**: `{index['number_of_scans']}`  
**Tuples Read**: `{index['tuples_read']}`  
**Tuples Fetched**: `{index['tuples_fetched']}`  
            """, unsafe_allow_html=False)

            st.markdown("**Definition:**")
            st.code(index["definition"], language="sql")

            delete_doc_btn = st.button(
                label="❌ Delete Index",
                key=f"delete_{index['name']}",
                on_click=delete_index,
                args=(index['name'], )
            )

    st.info("You've reached the end of the indices.")

def delete_index(name: str):
    response: requests.Response = requests.delete(
        url=f"http://r2r:7272/v3/indices/chunks/{name}",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Failed to delete index: {response.status_code} - {response.text}")
        return

    st.success(response.json()['results']['message'])

def create_index(file: UploadedFile):
    # Save uploaded file temporarily so as to read the data into dict
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".yaml")
    temp_file.write(file.getbuffer())
    temp_file.flush()

    try:
        index: Index = _load_index_config_from_yaml(temp_file.name)

        idx_config: Dict[str, Union[str, bool]] = _construct_index_config(
            index_method=index.method,
            index_name=index.name,
            index_measure=index.measure,
            index_arguments=index.arguments
        )

        response: requests.Response = requests.post(
            url="http://r2r:7272/v3/indices",
            headers={
                "Authorization": f"Bearer {st.session_state['bearer_token']}",
                "Content-Type": "application/json"
            },
            json={
                "config": idx_config
            },
            timeout=5
        )
        
        if response.status_code != 200:
            st.error(response.json()['detail']['message'])
            return
        
        st.success(response.json()['results']['message'])
    except ValueError as ve:
        st.error(f"Error in YAML file structure: {str(ve)}")
    finally:
        temp_file.close()


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
        data: Any = yaml.safe_load(f)

    if not isinstance(data, Dict) or len(data.keys()) != 1:
        raise ValueError(
            "YAML file must contain exactly one top-level key representing the index name!"
        )

    idx_name: str = list(data.keys())[0]
    config_data: Any = data[idx_name]

    if 'index_method' not in config_data or 'index_measure' not in config_data:
        raise ValueError(
            "The top-level key must contain 'index_method' and 'index_measure' fields."
        )

    idx_method: str = config_data['index_method']
    idx_measure: str = config_data['index_measure']
    idx_args: Dict = config_data.get('index_arguments', {})

    if not idx_name or not idx_method or not idx_measure:
        raise ValueError("YAML file must contain index_name, index_method, and index_measure.")

    return Index(idx_name, idx_method, idx_measure, idx_args)

def _construct_index_config(
    index_method: str,
    index_name: str,
    index_measure: str,
    index_arguments: dict
) -> Dict[str, Union[str, bool]]:
    if index_method not in ('hnsw', 'ivf_flat'):
        raise ValueError('Invalid index method, only hnsw and ivf_flat are supported!')

    if index_measure not in ('ip_distance', 'l2_distance', 'cosine_distance'):
        raise ValueError('Only ip_distance, l2_distance and cosine_distance are supported!')

    config: Dict[str, Union[str, bool]] = {
        # According to the documentation the table name should be vectors.
        # However, it actually needs to be on chunks.
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
