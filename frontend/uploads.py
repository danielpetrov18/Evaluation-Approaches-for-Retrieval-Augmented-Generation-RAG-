import os
import sys
import time
import tempfile
import pandas as pd
import streamlit as st
from pathlib import Path
from r2r import R2RException
from datetime import timedelta
from app import connect_to_backend
from langchain.docstore.document import Document

utility_dir = Path(__file__).parent.parent / 'backend' / 'utility'
sys.path.append(str(utility_dir)) 
from scraper import Scraper
from splitter import Splitter

st.markdown("""
    <style>
    .upload-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .file-list {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    .success-message {
        color: #28a745;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .error-message {
        color: #dc3545;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the loading process of files to improve performance. 
# In this case caching make sense, since if a user tries to upload the same file again, it would just be ignored.
# So for updating a particular file, the user should use the TO BE IMPLEMENTED button.
@st.cache_data(show_spinner=False, ttl=timedelta(minutes=60))
def save_uploaded_files(uploaded_files):
    """
    Save uploaded files to a temporary directory and return their paths.
    """
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    
    return saved_paths

# Makes no sense to use caching since this is a non-deterministic function.
def ingest_files(file_paths):
    """
    Ingest files using the R2R backend client.
    """
    try:
        client = connect_to_backend()
        with st.spinner("Ingesting files..."):
            client.ingest_files(file_paths)
        return True
    except R2RException as r2re:
        st.error(f"Error during ingestion: {str(r2re)}")
        return False
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return False

@st.cache_data(show_spinner=False, ttl=timedelta(minutes=60))
def extract_urls(file) -> list[str]:
    """
    Extract URLs from a CSV file without caching to ensure we always process
    the current file content.
    """
    if file is not None:
        filename = str(file.name)
        file_extension = filename.split('.')[1] # So if I have a urls.csv, I would get the extension part.

        if file_extension == 'csv':
            dataframe = pd.read_csv(file, header=None)  # Read the CSV with no header
            urls = dataframe.values.flatten()  # Flatten to get a 1D array of URLs
            urls = [url.strip() for url in urls if isinstance(url, str)]  # Clean up whitespace and non-strings
        else:
            raise ValueError(f"Unsupported or invalid file type: {file_extension}")
        
        return urls

@st.cache_data(show_spinner=False, ttl=timedelta(minutes=60))
def fetch_data_from_urls(urls: list[str]) -> list[Document]:
    return st.session_state.scraper.fetch_documents(urls) 

@st.cache_data(hash_funcs={Document: lambda x: x.metadata['source']}, show_spinner=False, ttl=timedelta(minutes=60))
def split_documents(documents: list[Document]) -> list[Document]:
    return st.session_state.splitter.split_documents(documents)

def ingest_chunks(urls: list[dict], split_documents: list[Document]):
    client = connect_to_backend()
    
    placeholder = st.empty()
    for url in urls:
        chunks = [split_doc for split_doc in split_documents if split_doc.metadata['source'] == url]
        
        if chunks:
            metadata = chunks[0].metadata
            chunks_text = [{"text": chunk.page_content} for chunk in chunks]
            try:
                resp = client.ingest_chunks(chunks_text, metadata)
                print(resp)
            except R2RException as r2re:
                print(f"Failed to ingest chunks for: [{metadata['source']}]! {str(r2re)}")
                placeholder = st.warning(f"Failed to ingest chunks for: [{metadata['source']}]! {str(r2re)}", icon="⚠️")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                placeholder = st.warning(f"Unexpected error: {str(e)}", icon="⚠️")
        else:
            print(f"No chunks found for [{url}]!")
            placeholder = st.warning(f"No chunks found for [{url}]!", icon="⚠️")
            continue 

st.title("📤 File Upload & Ingestion")

upload_file_tab, upload_url_tab = st.tabs(["Upload files", "Upload URLs"])
    
with upload_file_tab:
    with st.expander("ℹ️ About File Ingestion"):
        st.markdown("""
        - Files are processed and text data is being extracted
        - Files are then split into chunks for efficient retrieval
        - Finally, files are converted into embeddings using pgvector
        - Large files may take longer to process
        - Supported file formats: TXT, JSON, HTML, PDF, DOCX, PPTX, XLSX, CSV, MD
        - For best results, ensure documents are well-formatted
        """)
        # Initialize session states
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0

    # File upload section
    with st.container():
        st.markdown("### Upload Documents")
        st.markdown("Select one or multiple files to upload for ingestion into the system.")
        
        # Use a dynamic key for the file uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=["txt", "pdf", "doc", "docx", "md", "json", "html", "htm", "csv", "xlsx", "pptx"],
            help="Supported formats: TXT, PDF, DOC, DOCX, MD, JSON, HTML, CSV, XLSX, PPTX",
            key=f"file_uploader_{st.session_state.upload_key}"
        )
        
        # Update displayed files only when there are uploaded files
        if uploaded_files:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Start Ingestion", type="primary", use_container_width=True):
                    file_paths = save_uploaded_files(uploaded_files)
                    if ingest_files(file_paths):
                        st.success("Successful ingestion!", icon="✅")
                        st.session_state.upload_key += 1
                        time.sleep(2)
                        st.rerun()
            with col2:
                if st.button("Clear All", use_container_width=True):
                    st.session_state.upload_key += 1
                    st.rerun()

with upload_url_tab:
    with st.expander("ℹ️ About URL Ingestion"):
        st.markdown("""
        - Provide a csv file with a list of URLs
        - Make sure each URL is separated by a comma
        - Upload the file and click "Ingest data from web pages"
        """)
        
    if "upload_url_key" not in st.session_state:
        st.session_state.upload_url_key = 0
    if "scraper" not in st.session_state:
        st.session_state.scraper = Scraper()
    if "splitter" not in st.session_state:
        st.session_state.splitter = Splitter()

    with st.container():
        st.markdown("### Upload File")
        st.markdown("Select a single file containing URLs of web pages for ingestion.")
        
        uploaded_url_file = st.file_uploader(
            "Choose file containing URLs",
            type=["csv"],
            help="Supported formats: CSV",
            key=f"file_url_uploader_{st.session_state.upload_url_key}"
        )
        
        if uploaded_url_file:
            if st.button("Ingest data from web pages", type="primary", use_container_width=True):
                with st.status(label="Processing ...", expanded=True):
                    try:
                        urls = extract_urls(uploaded_url_file)
                    except ValueError as ve:
                        st.error(f"Error: {str(ve)}")
                        st.session_state.upload_url_key += 1
                        st.rerun()
                    st.write('Extracted URLs ...')
                    documents = fetch_data_from_urls(urls)
                    st.write('Fetched data ...')
                    split_documents = split_documents(documents)
                    st.write('Split data ...')
                    if split_documents:
                        ingest_chunks(urls, split_documents)
                        st.success("Successful ingestion!", icon="✅")
                        st.session_state.upload_key += 1
                        time.sleep(5)
                        st.rerun()
                    else:
                        st.warning("No data for ingestion found in the selected file! Make sure the URLs are valid!", icon="⚠️")
                        st.session_state.upload_url_key += 1
                        time.sleep(3)
                        st.rerun()