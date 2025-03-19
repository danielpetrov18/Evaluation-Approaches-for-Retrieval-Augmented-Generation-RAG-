"""Backend functionality for r2r."""

# pylint: disable=E0401
# pylint: disable=W0612
# pylint: disable=W0718
# pylint: disable=R1732

import os
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from r2r import (
    R2RException,
    R2RClient
)
import streamlit as st
from streamlit.errors import Error
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    BSHTMLLoader,
    AsyncHtmlLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

@st.cache_resource
def load_ascraper(urls: list[str]):
    """Load object to retrieve data from the internet"""
    return AsyncHtmlLoader(
        web_path=urls,
        default_parser="lxml"
    )

@st.cache_resource
def load_splitter():
    """Load RecursiveCharacterTextSplitter."""
    separators = ["\n\n", "\n"]

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size = st.session_state['chunk_size'],
        chunk_overlap = st.session_state['chunk_overlap'],
        length_function = len,
        separators = separators
    )

    return recursive_splitter

def fetch_documents(client: R2RClient, ids: list[str], offset: int, limit: int):
    """Retrieve documents"""
    try:
        selected_files = client.documents.list(ids, offset, limit).results
        if selected_files:
            for i, doc in enumerate(selected_files):
                with st.expander(label=f"Document: {doc.title}", expanded=False):
                    st.json(doc)

                    with st.popover(label="Delete document", icon="🗑️"):
                        delete_doc_btn = st.button(
                            label="Confirm deletion",
                            key=f"delete_document_{i}",
                            on_click=delete_document,
                            args=(client, doc.id, )
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
                label=f"{i}. Chunk {chunk.id}",
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
    """Creating a temp file and ingesting it"""
    try:
        # Save file temporarily and also save it with its proper title
        temp_dir = tempfile.gettempdir()
        temp_filepath = os.path.join(temp_dir, file.name)
        with open(file=temp_filepath, mode="wb") as temp_file:
            temp_file.write(file.getbuffer())

        with st.spinner(text="Ingesting document...", show_time=True):
            if metadata:
                metadata = json.loads(metadata)

            ingest_resp = client.documents.create(
                file_path=temp_filepath,
                ingestion_mode="custom",
                metadata=metadata,
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

    with st.status(
        label="Processing URLs...",
        expanded=True,
        state="running"
    ):
        try:
            urls = _extract_urls(file)
            if len(urls) > 0:
                st.write('Extracted URLs...')

                documents = _fetch_data_from_urls(urls)
                st.write('Fetched data...')

                for document in documents:
                    with tempfile.NamedTemporaryFile(delete=True, suffix=".html") as temp_file:
                        temp_file.write(document.page_content.encode('utf-8'))
                        temp_file.flush()
                        bs4_parser = BSHTMLLoader(temp_file.name)
                        splitted_documents = bs4_parser.load_and_split(load_splitter())
                        chunks_text = [chunk.page_content for chunk in splitted_documents]
                        try:
                            chunks_ing_resp = client.documents.create(
                                chunks=chunks_text,
                                ingestion_mode='custom',
                                metadata=document.metadata,
                                run_with_orchestration=True
                            ).results
                            st.success(chunks_ing_resp.message)
                        except R2RException as r2re:
                            st.error(f"Error {document.metadata['source']}: {str(r2re)}")
                st.info("Completed URL ingestion process")
            else:
                st.error("No valid URLs found in file")
        except R2RException as r2re:
            st.error(f"Error: {str(r2re)}")
        except ValueError as ve:
            st.error(f"Error: {str(ve)}")
        except Error as e:
            st.error(f"Unexpected streamlit error: {str(e)}")
        except Exception as exc:
            st.error(f"Unexpected error: {str(exc)}")

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
    return extracted_urls

def _run_async_function(coroutine):
    """Run an async function inside a synchronous Streamlit app."""
    return asyncio.run(coroutine)

def _remove_duplicate_urls(urls: list[str]) -> list[str]:
    return list(set(urls))
