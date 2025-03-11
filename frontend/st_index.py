import sys
import yaml
import asyncio
import streamlit as st
from pathlib import Path
from frontend.st_app import load_client
from r2r import R2RException

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from index import IndexHandler

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_async(coro):
    return loop.run_until_complete(coro)

st.title("📊 Index Management")

client = load_client()
index_handler = IndexHandler(client=client)

def load_index_config_from_yaml(filepath: str) -> tuple[str, str, str, dict, bool]:
    """
    Load index configuration from a YAML file.

    Expected structure:
    index_name:
      index_method: <hnsw|ivf_flat>
      index_measure: <ip_distance|l2_distance|cosine_distance>
      index_arguments: dict of arguments or empty
      concurrently: bool (optional, defaults to True)

    Returns:
        index_name, index_method, index_measure, index_arguments, concurrently
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or len(data.keys()) != 1:
        raise ValueError("YAML file must contain exactly one top-level key representing the index name!")

    index_name = list(data.keys())[0]
    config_data = data[index_name]

    if 'index_method' not in config_data or 'index_measure' not in config_data:
        raise ValueError("The top-level key must contain 'index_method' and 'index_measure' fields.")

    index_method = config_data['index_method']
    index_measure = config_data['index_measure']
    index_arguments = config_data.get('index_arguments', {})
    concurrently = config_data.get('concurrently', True)

    if not index_name or not index_method or not index_measure:
        raise ValueError("YAML file must contain index_name, index_method, and index_measure.")

    return index_name, index_method, index_measure, index_arguments, concurrently

tab_list, tab_create, tab_retrieve, tab_delete = st.tabs(["List Indices", "Create Index", "Retrieve Index", "Delete Index"])

with tab_list:
    st.subheader("List Indices")
    list_btn = st.button("Fetch Indices")
    if list_btn:
        try:
            indices = run_async(index_handler.list_indices())
            if indices:
                st.write(f"Found {len(indices)} indices:")
                st.json(indices)
            else:
                st.info("No indices found.")
        except R2RException as r2re:
            st.error(f"Error listing indices: {r2re}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

with tab_create:
    st.subheader("Create Index from YAML")
    st.write("Upload a YAML file that defines an index configuration. The file should have one top-level key with the index name, and 'index_method', 'index_measure', 'index_arguments', and optionally 'concurrently'.")

    uploaded_file = st.file_uploader(
        label="Upload YAML Index File", 
        type=["yaml", "yml"]
    )
    create_btn = st.button("Create Index")

    if create_btn and uploaded_file is not None:
        index_config_dir = Path(backend_dir) / "index"
        index_config_dir.mkdir(parents=True, exist_ok=True)

        # Construct the target file path
        target_path = index_config_dir / uploaded_file.name

        # Check if file already exists
        if target_path.exists():
            st.error("A file with this name already exists. Please rename your file and try again.", icon="⚠️")
        else:
            # Save the uploaded file
            with open(target_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Parse the config
                index_name, index_method, index_measure, index_arguments, concurrently = load_index_config_from_yaml(str(target_path))
                
                # Create the index
                result = run_async(index_handler.create_index(
                    index_method=index_method, 
                    index_name=index_name, 
                    index_measure=index_measure, 
                    index_arguments=index_arguments,
                    concurrently=concurrently
                ))
                st.success(f"Index created successfully! Message: {result}", icon="✅")
            except ValueError as ve:
                target_path.unlink(missing_ok=True)
                st.error(f"Error in YAML file structure: {ve}")
            except R2RException as r2re:
                target_path.unlink(missing_ok=True)
                st.error(f"Error creating index: {r2re}")
            except Exception as e:
                target_path.unlink(missing_ok=True)
                st.error(f"Unexpected error: {e}")

with tab_retrieve:
    st.subheader("Retrieve Index by Name")
    index_name_to_retrieve = st.text_input("Index Name to Retrieve")
    retrieve_btn = st.button("Get Index Details")

    if retrieve_btn and index_name_to_retrieve.strip():
        try:
            index_data = run_async(index_handler.get_index_details(index_name_to_retrieve.strip()))
            if index_data:
                st.write("Index Data:")
                st.json(index_data)
            else:
                st.info("Index not found.")
        except R2RException as r2re:
            st.error(f"Error retrieving index: {r2re}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

with tab_delete:
    st.subheader("Delete Index by Name")
    del_index_name = st.text_input("Index Name to Delete")
    del_btn = st.button("Delete Index")

    if del_btn and del_index_name.strip():
        try:
            result = run_async(
                index_handler.delete_index_by_name(
                    del_index_name.strip()
                )
            )
            
            index_dir = Path(backend_dir) / "index"
            for file in index_dir.iterdir():
                fullpath = str(index_dir / file.name)
                index_name, _, _, _, _ = load_index_config_from_yaml(fullpath)
                if index_name == del_index_name.strip():
                    file.unlink(missing_ok=True)
                    break
          
            st.success(f"Index deletion result: {result}", icon="🗑️")
        
        except R2RException as r2re:
            st.error(f"Error deleting index: {r2re}")
        
        except Exception as e:
            st.error(f"Unexpected error: {e}")
