"""GUI support for interacting with documents."""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import streamlit as st
from streamlit.errors import Error
from r2r import R2RException
from st_app import load_client
from langchain.docstore.document import Document

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from storage import Storage
from utility.ascrapper import AsyncScraper
from utility.splitter import Splitter

@st.cache_resource
def load_ascraper(url_list: list[str]):
    """Load the AsyncScraper with the provided URLs."""
    return AsyncScraper(url_list)

@st.cache_resource
def load_splitter():
    """Load RecursiveCharacterTextSplitter."""
    return Splitter()

@st.cache_resource
def load_storage_handler():
    """Get object for managing storage."""
    return Storage(client=load_client())

def extract_urls(file) -> list[str]:
    """Extracts URLs from a provided CSV file for web scrapping."""
    if file is None:
        raise FileNotFoundError("File not found")

    filename = str(file.name)
    file_extension = Path(filename).suffix.lower()

    if file_extension != ".csv":
        raise ValueError(f"Unsupported file type: {file_extension}")

    dataframe = pd.read_csv(
        filepath_or_buffer=file,
        usecols=[0],
        header=None
    )

    if dataframe.empty:
        raise ValueError("CSV file is empty")

    extracted_urls = dataframe.iloc[1:, 0].dropna().astype(str).str.strip().tolist()
    return extracted_urls

def run_async_function(coroutine):
    """Run an async function inside a synchronous Streamlit app."""
    return asyncio.run(coroutine)

def fetch_data_from_urls(scrape_urls: list[str]) -> list[Document]:
    """Fetches data from the provided URLs asynchronously."""
    ascraper = load_ascraper(scrape_urls)
    return run_async_function(ascraper.fetch_documents())

def display_documents(documents):
    """Better way to visualize documents"""
    for i, document in enumerate(documents):
        with st.expander(
            label=f"Document: {document.title} ({document.id})",
            expanded=False
        ):
            # Create two columns for metadata and content
            meta_col, content_col = st.columns([1, 2])

            with meta_col:
                st.markdown("### Metadata")
                st.markdown(f"**ID:** {document.id}")
                st.markdown(f"**Created:** {document.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
                           if hasattr(document, 'created_at') else "Date not available")

                if hasattr(document, 'metadata') and document.metadata:
                    st.json(document.metadata)
            
            # with content_col:
            #     st.markdown("### Content")
                
            #     # Show document content if available
            #     if hasattr(document, 'content') and document.content:
            #         st.text_area("Document Content", document.content, height=200, disabled=True)
            #     else:
            #         # Otherwise show the full JSON
            #         st.json(document)

            # Add action buttons for this document
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button(f"View Details #{document.id}", key=f"view_{document.id}"):
                    st.session_state["selected_document"] = document.id
            with action_col2:
                if st.button(f"Delete #{document.id}", key=f"delete_{document.id}", type="secondary"):
                    st.warning(f"Are you sure you want to delete document {document.id}?")
                    if st.button(f"Confirm Delete #{document.id}", key=f"confirm_delete_{document.id}", type="primary"):
                        try:
                            load_storage_handler().delete_document_by_id(document.id)
                            st.success(f"Document {document.id} deleted successfully!")
                        except R2RException as r2re:
                            st.error(f"Error deleting: {str(r2re)}")
                        except Error as e:
                            st.error(str(e))

# async def clear_files():
#     # Delete from vector store
#     try:
#         await storage.clean_db()
#     except Exception as e:
#         st.error(f"Failed to clear vector store: {e}")
#         return False
        
#     # Clear filesystem
#     files_dir = Path(FILES_DIRECTORY)
#     if files_dir.exists():
#         for file in files_dir.rglob('*'):
#             if file.is_file():
#                 file.unlink()
                
#     return True

if __name__ == "__page__":
    st.title("📄 Document Management")

    FILES_DIRECTORY = os.getenv("FILES_DIRECTORY")
    EXPORT_DIRECTORY = os.getenv("EXPORT_DIRECTORY")

    storage = load_storage_handler()

    t_list, t_retrieve, t_file_ingest, t_webscrape, t_export_docs, t_download  = st.tabs(
        [
            "List Documents", 
            "Retrieve Document",
            "Ingest File",
            "Webscrape URLs",
            "Export Documents",
            "Download Documents"
        ]
    )

    with t_list:
        st.markdown("**List Documents**")

        col1, col2 = st.columns([1, 2])
        with col1:
            offset = st.number_input("Offset", min_value=0, value=0)
        with col2:
            limit = st.number_input("Limit", min_value=1, max_value=100, value=10)

        if st.button("Fetch Documents", type="primary"):
            try:
                documents = storage.list_documents(offset=offset, limit=limit).results
                if documents:
                    for i, document in enumerate(documents):
                        with st.expander(
                            label=f"Document: {document.title} ({document.id})",
                            expanded=False
                        ):
                            st.json(document)

                    # Show a message if we've reached the end
                    if len(documents) < limit:
                        st.info("You've reached the end of the conversations list.")
                else:
                    st.info("No documents found.")
            except R2RException as r2re:
                st.error(f"Error: {str(r2re)}")
            except Error as e:
                st.error(f"Error: {str(e)}")

    with t_retrieve:
        st.markdown("**Retrieve Document**")
        doc_id = st.text_input(
            label="Document ID",
            placeholder="Ex. document_id"
        )

        if st.button("Get Document", type="primary"):
            if not doc_id:
                st.warning("Please enter a document ID")
            else:
                try:
                    document = storage.get_document_metadata_by_id(doc_id).results
                    st.json(document)
                except R2RException as r2re:
                    st.error(f"Error: {str(r2re)}")
                except Error as e:
                    st.error(f"Error: {str(e)}")

    # with tab_chunks:
    #     st.subheader("Document Chunks")
    #     chunk_doc_id = st.text_input("Document ID for chunks")
    #     col1, col2, col3 = st.columns([1, 1, 1])
    #     with col1:
    #         chunk_offset = st.number_input("Chunk Offset", min_value=0, value=0)
    #     with col2:
    #         chunk_limit = st.number_input("Chunk Limit", min_value=1, value=100)
    #     with col3:
    #         include_vectors = st.checkbox("Include Vectors")
        
    #     if st.button("Get Chunks", type="primary"):
    #         if chunk_doc_id:
    #             try:
    #                 chunks = run_async(storage.fetch_document_chunks(
    #                     doc_id=chunk_doc_id,
    #                     offset=chunk_offset,
    #                     limit=chunk_limit,
    #                     include_vectors=include_vectors
    #                 ))
    #                 st.json(chunks)
    #             except R2RException as r2re:
    #                 st.error(f"Error: {str(r2re)}")
    #             except Exception as e:
    #                 st.error(f"Error: {str(e)}")
    #         else:
    #             st.warning("Please enter a document ID")

    # with tab_chunk_retrieve:
    #     st.subheader("Retrieve Chunk")
    #     chunk_id = st.text_input("Chunk ID")
    #     if st.button("Get Chunk", type="primary"):
    #         if chunk_id:
    #             try:
    #                 chunk = run_async(storage.retrieve_chunk_by_id(chunk_id))
    #                 st.json(chunk)
    #             except R2RException as r2re:
    #                 st.error(f"Error: {str(r2re)}")
    #             except Exception as e:
    #                 st.error(f"Error: {str(e)}")
    #         else:
    #             st.warning("Please enter a chunk ID")

    with t_file_ingest:
        st.markdown("**Ingest Document**")

        uploaded_file = st.file_uploader(
            "Choose a file to upload",
            type=["txt", "pdf", "docx", "csv", "json", "md", "html"]
        )

        metadata = st.text_area(
            "Metadata (JSON format)",
            value="{}",
            help="Optional metadata in JSON format"
        )

        if st.button("Ingest Document", type="primary"):
            try:
                file_path = Path(FILES_DIRECTORY) / uploaded_file.name
                file_path.parent.mkdir(exist_ok=True, parents=True)

                if file_path.exists():
                    st.error(
                        body=f"A file with the name '{uploaded_file.name}' already exists.",
                        icon="⚠️"
                    )
                else:
                    with open(file=file_path, mode="wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner(text="Ingesting document...", show_time=True):
                        metadata = json.loads(metadata)
                        ingest_resp = storage.ingest_file(str(file_path), metadata).results
                        st.success(ingest_resp.message)
            except json.JSONDecodeError as jde:
                file_path.unlink()
                st.error("Error: Invalid JSON format in file.")
            except R2RException as r2re:
                file_path.unlink()
                st.error(f"Error: {str(r2re)}")
            except Error as e:
                file_path.unlink()
                st.error(f"Error: {str(e)}")

    with t_webscrape:
        st.markdown("**Perform Web Scrape**")

        uploaded_url_file = st.file_uploader(
            label="Choose file containing URLs",
            type="csv",
            help="Supported formats: CSV"
        )

        if st.button("Ingest data from URLs", type="primary"):
            with st.status(label="Processing URLs...", expanded=True, state="running"):
                try:
                    urls = extract_urls(uploaded_url_file)
                    if len(urls) > 0:
                        st.write('Extracted URLs...')

                        documents = fetch_data_from_urls(urls)
                        st.write('Fetched data...')

                        split_docs = load_splitter().split_documents(documents)
                        st.write('Split data...')

                        if split_docs:
                            for url in urls:
                                chunks = [d for d in split_docs if d.metadata['source'] == url]
                                if chunks:
                                    metadata = chunks[0].metadata
                                    chunks_text = [chunk.page_content for chunk in chunks]
                                    try:
                                        chunks_ing_resp = storage.ingest_chunks(
                                            chunks_text,
                                            metadata
                                        ).results
                                        st.success(chunks_ing_resp.message)
                                    except R2RException as r2re:
                                        st.error(f"Error {url}: {str(r2re)}")
                            st.info("Completed URL ingestion process")
                        else:
                            st.warning("No valid content found in the provided URLs")
                    else:
                        st.error("No valid URLs found in file")
                except R2RException as r2re:
                    st.error(f"Error: {str(r2re)}")
                except ValueError as ve:
                    st.error(f"Error: {str(ve)}")

    with t_export_docs:
        st.markdown("**Export Documents**")

        files_out = st.text_input(
            label='Name of output file (without extension)',
            placeholder="Ex. exported_docs"
        )

        if st.button("Export Documents", type="primary"):
            try:
                storage.export_documents_to_csv(
                    out_path = files_out,
                    bearer_token = st.session_state["token"]
                )
                st.success("Successfully exported documents!")
            except R2RException as r2re:
                st.error(f"Error: {str(r2re)}")
            except Error as e:
                st.error(f"Error: {str(e)}")

    with t_download:
        st.markdown("**Download Documents**")

        download_out = st.text_input(
            label='Name of zip file to download',
            placeholder="Ex. documents"
        )

        col1, col2 = st.columns(2)

        with col1:
            use_date_filter = st.checkbox("Filter by date range")

            if use_date_filter:
                start_date_filter = st.date_input(
                    label="Start date",
                    value=datetime(2025, 1, 1)
                )

                end_date_filter = st.date_input(
                    label="End date",
                    value=datetime.now()
                )

        with col2:
            use_id_filter = st.checkbox("Filter by document IDs")

            if use_id_filter:
                document_ids_input = st.text_area(
                    label="Document IDs (one per line)",
                    placeholder="Enter document IDs, one per line"
                )

        # Provide the ability to filter based on ids, start and end date

        if st.button("Download Documents", type="primary"):
            if not download_out:
                st.error("Please provide a name for the ZIP file.")
            else:
                try:
                    document_ids = None
                    if use_id_filter and document_ids_input:
                        document_ids = [doc_id.strip() for doc_id in document_ids_input.split('\n') if doc_id.strip()]
      
                    # Process date filters if enabled
                    start_date = None
                    end_date = None
                    if use_date_filter:
                        # Convert date to datetime at beginning and end of day
                        start_date = datetime.combine(start_date_filter, datetime.min.time())
                        end_date = datetime.combine(end_date_filter, datetime.max.time())

                    with st.spinner("Preparing documents for download..."):
                        storage.export_documents_to_zip(
                            out_path=download_out,
                            bearer_token=st.session_state["token"],
                            document_ids=document_ids,
                            start_date=start_date,
                            end_date=end_date
                        )

                    st.success(f"Successfully downloaded documents!")
                except R2RException as r2re:
                    st.error(f"Error: {str(r2re)}")
                except Error as e:
                    st.error(f"Error: {str(e)}")

    # with tab_delete:
    #     st.subheader("Delete Document")
    #     delete_tabs = st.tabs(["Delete by ID", "Delete by Filter"])
        
    #     with delete_tabs[0]:
    #         del_doc_id = st.text_input("Document ID to delete")
    #         if st.button("Delete Document", type="primary"):
    #             if del_doc_id:
    #                 try:
    #                     result = run_async(storage.delete_document_by_id(del_doc_id))
    #                     st.success("Document deleted successfully!")
    #                     st.json(result)
    #                 except Exception as e:
    #                     st.error(f"Error: {str(e)}")
    #             else:
    #                 st.warning("Please enter a document ID")
        
    #     with delete_tabs[1]:
    #         filter_json = st.text_area(
    #             "Filter (JSON format)",
    #             value="{}",
    #             help="Filter criteria in JSON format"
    #         )
    #         if st.button("Delete Documents", type="primary"):
    #             try:
    #                 filters = json.loads(filter_json)
    #                 result = run_async(storage.delete_documents_by_filter(filters))
    #                 st.success("Documents deleted successfully!")
    #                 st.json(result)
    #             except R2RException as r2re:
    #                 st.error(f"Error: {str(r2re)}")
    #             except Exception as e:
    #                 st.error(f"Error: {str(e)}")

    # with tab_clear:
    #     st.subheader("Clear All Files")
    #     if st.button("Clear Files", type="primary"):
    #         if run_async(clear_files()):
    #             st.success("All files cleared successfully!")
    #         else:
    #             st.error("Files directory not found")