import sys
import json
import asyncio
import pandas as pd
import streamlit as st
from pathlib import Path
from app import load_client
from r2r import R2RException
from langchain.docstore.document import Document

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from storage import StorageHandler
from utility.scraper import Scraper
from utility.splitter import Splitter

FILES_DIRECTORY = backend_dir / "data"

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

@st.cache_resource
def load_scraper():
    return Scraper()

@st.cache_resource
def load_splitter():
    return Splitter()

def run_async(coro):
    return loop.run_until_complete(coro)

def extract_urls(file) -> list[str]:
    if file is not None:
        filename = str(file.name)
        file_extension = filename.split('.')[1]
        if file_extension == 'csv':
            dataframe = pd.read_csv(file, header=None)
            urls = dataframe.values.flatten()
            return [url.strip() for url in urls if isinstance(url, str)]
        raise ValueError(f"Unsupported file type: {file_extension}")
    return []

def fetch_data_from_urls(urls: list[str]) -> list[Document]:
    scraper = load_scraper()
    return scraper.fetch_documents(urls)

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = load_splitter()
    return splitter.split_documents(documents)

async def clear_files():
    # Delete from vector store
    try:
        await storage.clean_db()
    except Exception as e:
        st.error(f"Failed to clear vector store: {e}")
        return False
        
    # Clear filesystem
    files_dir = Path(FILES_DIRECTORY)
    if files_dir.exists():
        for file in files_dir.rglob('*'):
            if file.is_file():
                file.unlink()
                
    return True

st.title("📄 Document Management")

client = load_client()
storage = StorageHandler(client=client)

tab_list, tab_retrieve, tab_chunks, tab_chunk_retrieve, tab_ingest, tab_chunk_ingest, tab_delete, tab_clear = st.tabs(
    [
        "List Documents", 
        "Retrieve Document", 
        "Document Chunks", 
        "Retrieve Chunk",
        "Ingest Document", 
        "Ingest Chunks", 
        "Delete Document",
        "Clear Files"
    ]
)

with tab_list:
    st.subheader("List Documents")
    col1, col2 = st.columns([1, 2])
    with col1:
        offset = st.number_input("Offset", min_value=0, value=0)
    with col2:
        limit = st.number_input("Limit", min_value=1, max_value=100, value=10)
        
    if st.button("Fetch Documents", type="primary"):
        try:
            documents = run_async(storage.list_documents(offset=offset, limit=limit))
            if documents:
                st.write(f"Found {len(documents)} documents:")
                st.json(documents)
            else:
                st.info("No documents found.")
        except R2RException as r2re:
            st.error(f"Error: {str(r2re)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab_retrieve:
    st.subheader("Retrieve Document")
    doc_id = st.text_input("Document ID")
    
    if st.button("Get Document", type="primary"):
        if doc_id:
            try:
                document = run_async(storage.retrieve_document_by_id(doc_id))
                st.json(document)
            except R2RException as r2re:
                st.error(f"Error: {str(r2re)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a document ID")

with tab_chunks:
    st.subheader("Document Chunks")
    chunk_doc_id = st.text_input("Document ID for chunks")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        chunk_offset = st.number_input("Chunk Offset", min_value=0, value=0)
    with col2:
        chunk_limit = st.number_input("Chunk Limit", min_value=1, value=100)
    with col3:
        include_vectors = st.checkbox("Include Vectors")
    
    if st.button("Get Chunks", type="primary"):
        if chunk_doc_id:
            try:
                chunks = run_async(storage.fetch_document_chunks(
                    doc_id=chunk_doc_id,
                    offset=chunk_offset,
                    limit=chunk_limit,
                    include_vectors=include_vectors
                ))
                st.json(chunks)
            except R2RException as r2re:
                st.error(f"Error: {str(r2re)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a document ID")

with tab_chunk_retrieve:
    st.subheader("Retrieve Chunk")
    chunk_id = st.text_input("Chunk ID")
    if st.button("Get Chunk", type="primary"):
        if chunk_id:
            try:
                chunk = run_async(storage.retrieve_chunk_by_id(chunk_id))
                st.json(chunk)
            except R2RException as r2re:
                st.error(f"Error: {str(r2re)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a chunk ID")

with tab_ingest:
    st.subheader("Ingest Document")
    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=["txt", "pdf", "docx", "csv", "json", "md"]
    )
    metadata = st.text_area(
        "Metadata (JSON format)",
        value="{}",
        help="Optional metadata in JSON format"
    )
    
    if st.button("Ingest Document", type="primary"):
        if uploaded_file:
            try:
                file_path = Path(FILES_DIRECTORY) / uploaded_file.name
                file_path.parent.mkdir(exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                with st.spinner("Ingesting document..."):
                    result = run_async(
                                asyncio.wait_for(
                                    storage.ingest_file(
                                        filepath=str(file_path),
                                        metadata=json.loads(metadata)
                                    ),
                                    timeout=1200  # 20 minute timeout
                                )
                             )
                    st.success("Document ingested successfully!")
                    st.json(result)
            except R2RException as r2re:
                st.error(f"Error: {str(r2re)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if file_path.exists():
                    file_path.unlink()
        else:
            st.warning("Please upload a file")

with tab_chunk_ingest:
    st.subheader("Ingest from URLs")
    uploaded_url_file = st.file_uploader(
        label="Choose file containing URLs",
        type="csv",
        help="Supported formats: CSV"
    )

    if st.button("Ingest data from URLs", type="primary"):
        if uploaded_url_file:
            with st.status(label="Processing URLs...", expanded=True, state="running"):
                try:
                    urls = extract_urls(uploaded_url_file)
                    if len(urls) > 0:
                        st.write('Extracted URLs...')
                        
                        documents = fetch_data_from_urls(urls)
                        st.write('Fetched data...')
                        
                        split_docs = split_documents(documents)
                        st.write('Split data...')
                        
                        if split_docs:
                            for url in urls:
                                chunks = [doc for doc in split_docs if doc.metadata['source'] == url]
                                if chunks:
                                    metadata = chunks[0].metadata
                                    chunks_text = [chunk.page_content for chunk in chunks]
                                    try:
                                        result = run_async(
                                            asyncio.wait_for(
                                                storage.ingest_chunks(
                                                    chunks_text, metadata
                                                ),
                                                timeout=3600  # 1 hour timeout, since all the urls will be ingested together, might take a while
                                            )
                                        )
                                        if "error" not in result:
                                            st.success(f"Successfully ingested chunks from {url}")
                                        else:
                                            st.error(f"Failed to ingest chunks from {url}: {result['error']}")
                                    except Exception as e:
                                        st.error(f"Unexpected error processing {url}: {str(e)}")
                            st.success("Completed URL ingestion process")
                        else:
                            st.warning("No valid content found in the provided URLs")
                    else:
                        st.error("No valid URLs found in file")
                except R2RException as r2re:
                    st.error(f"Error: {str(r2re)}")
                except ValueError as ve:
                    st.error(f"Error: {str(ve)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
        else:
            st.warning("Please upload a CSV file with URLs")

with tab_delete:
    st.subheader("Delete Document")
    delete_tabs = st.tabs(["Delete by ID", "Delete by Filter"])
    
    with delete_tabs[0]:
        del_doc_id = st.text_input("Document ID to delete")
        if st.button("Delete Document", type="primary"):
            if del_doc_id:
                try:
                    result = run_async(storage.delete_document_by_id(del_doc_id))
                    st.success("Document deleted successfully!")
                    st.json(result)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a document ID")
    
    with delete_tabs[1]:
        filter_json = st.text_area(
            "Filter (JSON format)",
            value="{}",
            help="Filter criteria in JSON format"
        )
        if st.button("Delete Documents", type="primary"):
            try:
                filters = json.loads(filter_json)
                result = run_async(storage.delete_documents_by_filter(filters))
                st.success("Documents deleted successfully!")
                st.json(result)
            except R2RException as r2re:
                st.error(f"Error: {str(r2re)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab_clear:
    st.subheader("Clear All Files")
    if st.button("Clear Files", type="primary"):
        if run_async(clear_files()):
            st.success("All files cleared successfully!")
        else:
            st.error("Files directory not found")