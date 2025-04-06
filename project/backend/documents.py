"""Backend functionality for r2r."""

# pylint: disable=E0401
# pylint: disable=W0612
# pylint: disable=W0718
# pylint: disable=R0914
# pylint: disable=R1732

import os
import time
import asyncio
import tempfile
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json
import pandas as pd
import streamlit as st
from r2r import R2RException, R2RClient

from streamlit.errors import Error
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain.docstore.document import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import BSHTMLLoader, AsyncHtmlLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

@st.cache_resource
def load_unstructured(filepath: str, ingestion_conf: Dict[str, Any]):
    """Load object to to extract text from files."""
    return UnstructuredFileLoader(
        file_path=filepath,
        mode="single",
        unstructured_kwargs=ingestion_conf
    )

@st.cache_resource
def load_token_splitter():
    """Load TokenTextSplitter."""
    return TokenTextSplitter(
        chunk_size = st.session_state['chunk_size'],
        chunk_overlap = st.session_state['chunk_overlap']
    )

@st.cache_resource
def load_ascraper(urls: list[str]):
    """Load object to retrieve data from the internet"""
    return AsyncHtmlLoader(
        web_path=urls,
        default_parser="lxml"
    )

def fetch_documents(client: R2RClient, ids: list[str], offset: int, limit: int):
    """Retrieve documents. For each document there's a delete and update metadata buttons."""
    try:
        selected_files = client.documents.list(ids, offset, limit).results
        if selected_files:
            for i, doc in enumerate(selected_files):
                with st.expander(label=f"{i + 1}: {doc.metadata['filename']}", expanded=False):
                    st.json(doc)

                    delete_col, update_metadata_col = st.columns(2)

                    with delete_col:
                        with st.popover(label="Delete document", icon="🗑️"):
                            delete_doc_btn = st.button(
                                label="Confirm deletion",
                                key=f"delete_document_{i}",
                                on_click=delete_document,
                                args=(client, doc.id, )
                            )

                    with update_metadata_col:
                        with st.popover(label="Update metadata", icon="✏️"):
                            updated_metadata_key = str(uuid4())
                            metadata_str = st.text_area(
                                label="Update metadata",
                                key=updated_metadata_key,
                                value=json.dumps(doc.metadata, indent=4),
                                height=150
                            )

                            update_metadata_btn = st.button(
                                label="Confirm update",
                                key=f"update_metadata_btn_{i}",
                                on_click=update_metadata,
                                args=(client, doc.id, updated_metadata_key)
                            )

            if len(selected_files) < limit:
                st.info("You've reached the end of the documents.")
        else:
            st.info("No documents found.")
    except R2RException as r2re:
        st.error(f"Error when fetching documents: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def delete_document(client: R2RClient, document_id: str):
    """Delete any document by id"""
    try:
        client.documents.delete(document_id)
        st.success(f"Successfully deleted document: {document_id}")
    except R2RException as r2re:
        st.error(f"Error when deleting document: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def update_metadata(client: R2RClient, document_id: str, updated_metadata_key: str):
    """Update or add fields to metadata of a document"""
    try:
        metadata = st.session_state[updated_metadata_key]

        key_value_pairs = []
        if metadata:
            metadata = json.loads(metadata)
            key_value_pairs = [{k: v} for k, v in metadata.items()]

        client.documents.append_metadata(
            id=document_id,
            metadata=key_value_pairs
        )
        st.success("Successfully updated metadata")
    except json.JSONDecodeError as jde:
        st.error(f"Error: {str(jde)}")
    except R2RException as r2re:
        st.error(f"Error: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def fetch_document_chunks(client: R2RClient, document_id: str, offset: int, limit: int):
    """Fetches all chunks related to a document"""
    try:
        chunks = client.documents.list_chunks(
            id=document_id,
            include_vectors=False,
            offset=offset,
            limit=limit
        ).results

        for i, chunk in enumerate(chunks):
            with st.expander(
                label=f"Chunk {i + 1}: {chunk.id}",
                expanded=False
            ):
                st.markdown("### Text: \n", unsafe_allow_html=True)
                st.markdown(chunk.text)
                st.markdown("### Metadata: \n", unsafe_allow_html=True)
                for k, v in chunk.metadata.items():
                    st.markdown(f"* **{k.upper()}**: `{v}`")

    except R2RException as r2re:
        st.error(f"Error when fetching chunks for document: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def ingest_file(client: R2RClient, file: UploadedFile, metadata: dict):
    """
    This method takes a file in binary format. Saves it in the tmp folder.
    Thereafter, the text is extracted and split using token splitter.
    Finally, it gets ingested. Alternatively, one can ingest the document
    directly, however I prefer the token splitter since the chunks make sense to me.
    If you use the R2R unstructured service, you can ingest directly.
    """
    # Do it outside because of the finally clause
    temp_filepath = os.path.join(tempfile.gettempdir(), file.name)
    try:
        # Make sure file doesn't already exist.
        for doc in client.documents.list().results:
            if doc.metadata["filename"] == file.name:
                st.error("File already exists!")
                return

        with open(file=temp_filepath, mode="wb") as temp_file:
            temp_file.write(file.getbuffer())

        if not os.path.exists(temp_filepath) or os.path.getsize(temp_filepath) == 0:
            st.error("Failed to save file or file is empty.")
            return

        loader = load_unstructured(
            filepath=temp_filepath,
            ingestion_conf=st.session_state['ingestion_config']
        )
        document: Document = loader.load()[0]

        splitter = load_token_splitter()
        chunks: List[Document] = splitter.split_documents([document])
        txt_chunks: List[str] = [chunk.page_content for chunk in chunks]

        with st.spinner(text="Ingesting document...", show_time=True):
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            combined_metadata = {**document.metadata, **metadata}
            combined_metadata["filename"] = file.name

            ingest_resp = client.documents.create(
                chunks=txt_chunks,
                ingestion_mode="fast",
                metadata=combined_metadata,
                run_with_orchestration=True
            ).results
            st.success(ingest_resp.message)
    except json.JSONDecodeError as jde:
        st.error(f"Error: {str(jde)}")
    except R2RException as r2re:
        st.error(f"Error: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")
    finally:
        # Remove temporary file after ingestion
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

def perform_webscrape(client: R2RClient, file: UploadedFile):
    """By providing a file with URLs we can scrape and ingest the data"""

    # with st.status(
    #     label="Processing URLs...",
    #     expanded=True,
    #     state="running"
    # ):
    #     try:
    #         urls = _extract_urls(file)
    #         if len(urls) > 0:
    #             st.write('Extracted URLs...')

    #             documents = _fetch_data_from_urls(urls)
    #             st.write('Fetched data...')

    #             for document in documents:
    #                 with tempfile.NamedTemporaryFile(delete=True, suffix=".html") as temp_file:
    #                     temp_file.write(document.page_content.encode('utf-8'))
    #                     temp_file.flush()
    #                     bs4_parser = BSHTMLLoader(temp_file.name)
    #                     splitted_documents = bs4_parser.load_and_split(load_splitter())
    #                     chunks_text = [chunk.page_content for chunk in splitted_documents]
    #                     try:
    #                         chunks_ing_resp = client.documents.create(
    #                             chunks=chunks_text,
    #                             ingestion_mode='custom',
    #                             metadata=document.metadata,
    #                             run_with_orchestration=True
    #                         ).results
    #                         st.success(f"{document.metadata['source']}: {chunks_ing_resp.message}")
    #                     except R2RException as r2re:
    #                         st.error(f"Error {document.metadata['source']}: {str(r2re)}")
    #                     time.sleep(10) # Wait for ingestion
    #             st.info("Completed URL ingestion process")
    #         else:
    #             st.error("No valid URLs found in file")
    #     except R2RException as r2re:
    #         st.error(f"Error: {str(r2re)}")
    #     except ValueError as ve:
    #         st.error(f"Error: {str(ve)}")
    #     except Error as e:
    #         st.error(f"Unexpected streamlit error: {str(e)}")
    #     except Exception as exc:
    #         st.error(f"Unexpected error: {str(exc)}")

def export_docs_to_csv(client: R2RClient, filename: str, filetype: str, ingestion_status: str):
    """Exports all available documents to a csv file."""
    try:
        filters = {}

        if filetype != "all":
            filters["type"] = filetype

        if ingestion_status != "all":
            filters["ingestion_status"] = ingestion_status

        columns = [
            "id",
            "type",
            "title",
            "ingestion_status",
            "created_at",
            "updated_at",
            "summary",
            "total_tokens"
        ]

        out_path = Path(st.session_state['exports_dir']) / f"{filename}.csv"
        client.documents.export(
            output_path=out_path,
            columns=columns,
            filters=filters,
            include_header=True
        )
        st.success("Successfully exported documents!")
    except R2RException as r2re:
        st.error(f"Error: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def download_documents(
    client: R2RClient,
    download_out: str,
    document_ids: str = None,
    start_date_filter: datetime = None,
    end_date_filter: datetime = None
):
    """
        Downloads the files in the specified filepath.
        # Note: Only files that were ingested as a whole can be downloaded. NO chunks
    """
    try:
        if document_ids:
            document_ids = [doc_id.strip() for doc_id in document_ids.split('\n')]

        if start_date_filter and end_date_filter:
            # Convert date to datetime at beginning and end of day
            start_date_filter = datetime.combine(start_date_filter, datetime.min.time())
            end_date_filter = datetime.combine(end_date_filter, datetime.max.time())

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = Path(st.session_state['exports_dir']) / f"{download_out}_{timestamp}.zip"
        with st.spinner("Preparing documents for download..."):
            client.documents.download_zip(
                document_ids=document_ids,
                start_date=start_date_filter,
                end_date=end_date_filter,
                output_path=out_path
            )

        st.success("Successfully downloaded documents!")
    except R2RException as r2re:
        st.error(f"Error: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def _fetch_data_from_urls(scrape_urls: list[str]) -> list[Document]:
    """Fetches data from the provided URLs asynchronously."""
    ascraper = load_ascraper(scrape_urls)
    web_documents = _run_async_function(ascraper.aload())
    return web_documents

def _extract_urls(file: UploadedFile) -> list[str]:
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
    extracted_urls = _remove_duplicate_urls(extracted_urls)
    return extracted_urls

def _run_async_function(coroutine):
    """Run an async function inside a synchronous Streamlit app."""
    return asyncio.run(coroutine)

def _remove_duplicate_urls(urls: list[str]) -> list[str]:
    return list(set(urls))
