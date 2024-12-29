import sys
import yaml
import asyncio
import streamlit as st
from pathlib import Path
from app import load_client
from r2r import R2RException

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from prompt import PromptHandler 

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_async(coro):
    return loop.run_until_complete(coro)

st.title("📝 Prompt Management")

client = load_client()
prompt_handler = PromptHandler(client=client)

tab_list, tab_create, tab_retrieve, tab_delete = st.tabs(["List Prompts", "Create Prompt", "Retrieve Prompt", "Delete Prompt"])

with tab_list:
    st.subheader("List Prompts")
    list_btn = st.button("Fetch Prompt List")
    if list_btn:
        try:
            prompts = run_async(prompt_handler.list_prompts())
            if prompts:
                st.write(f"Found {len(prompts)} prompts:")
                st.json(prompts)  
            else:
                st.info("No prompts found.")
        except R2RException as r2re:
            st.error(f"Error listing prompts: {r2re}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

with tab_create:
    st.subheader("Create Prompt from YAML")
    st.write("Upload a YAML file that defines a prompt. The file should have one top-level key with the prompt name, and 'template' and 'input_types'.")

    uploaded_file = st.file_uploader(
        label="Upload YAML Prompt File", 
        type=["yaml", "yml"]
    )
    create_btn = st.button("Create Prompt")

    if create_btn and uploaded_file is not None:
        template_dir = Path(backend_dir) / "template"
        template_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = template_dir / uploaded_file.name
        if target_path.exists():
            st.error("A file with this name already exists. Please rename your file and try again.", icon="⚠️")
        else:    
            with open(target_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

            with open(target_path, 'r') as f:
                data = yaml.safe_load(f)
            
            index_name = list(data.keys())[0] # The key is top-level component
            
            try:
                run_async(prompt_handler.get_prompt_by_name(index_name)) # Check if prompt already exists
                st.error(f"A prompt with name '{index_name}' already exists. Please choose a different name.", icon="⚠️")
                target_path.unlink(missing_ok=True)
            except R2RException as r2re: # If prompt doesn't exist, create it
                try:
                    result = run_async(prompt_handler.create_prompt(str(target_path)))
                    st.success(f"Prompt created successfully! Message: {result}", icon="✅")
                except R2RException as r2re:
                    st.error(f"Error creating prompt: {r2re}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")  

with tab_retrieve:
    st.subheader("Retrieve Prompt by Name")
    prompt_name = st.text_input("Prompt Name to Retrieve")
    retrieve_btn = st.button("Get Prompt")

    if retrieve_btn and prompt_name.strip():
        try:
            prompt_data = run_async(prompt_handler.get_prompt_by_name(prompt_name.strip()))
            if prompt_data:
                st.write("Prompt Data:")
                st.json(prompt_data)
            else:
                st.info("Prompt not found.")
        except R2RException as r2re:
            st.error(f"Error retrieving prompt: {r2re}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

with tab_delete:
    st.subheader("Delete Prompt by Name")
    del_prompt_name = st.text_input("Prompt Name to Delete")
    del_btn = st.button("Delete Prompt")

    if del_btn and del_prompt_name.strip():
        try:
            result = run_async(
                prompt_handler.delete_prompt_by_name(
                    del_prompt_name.strip()
                )
            )
            
            prompt_dir = Path(backend_dir) / "template"
            for file in prompt_dir.iterdir():
                fullpath = str(prompt_dir / file.name)
                
                with open(fullpath, 'r') as f:
                    data = yaml.safe_load(f)
                
                prompt_name = list(data.keys())[0]
                if prompt_name == del_prompt_name.strip():
                    file.unlink(missing_ok=True)
                    break
            
            st.success(f"Prompt deletion result: {result}", icon="🗑️")
        except R2RException as r2re:
            st.error(f"Error deleting prompt: {r2re}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")