"""GUI support for interacting with documents."""

import asyncio
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import streamlit as st
from streamlit.errors import Error
from r2r import R2RException
from langchain.docstore.document import Document
from st_app import load_client # pylint: disable=E0401
from utility.splitter import Splitter # pylint: disable=E0401
from utility.ascrapper import AsyncScraper # pylint: disable=E0401

@st.cache_resource
def load_ascraper(url_list: list[str]):
    """Load the AsyncScraper with the provided URLs."""
    return AsyncScraper(url_list)

@st.cache_resource
def load_splitter():
    """Load RecursiveCharacterTextSplitter."""
    return Splitter()

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

def delete_document(document_id: str):
    """Delete any document by id"""
    try:
        load_client().documents.delete(document_id)
        st.success(f"Successfully deleted document: {document_id}")
    except R2RException as r2re:
        st.error(f"Error: {str(r2re)}")
    except Error as e:
        st.error(f"Error: {str(e)}")

def fetch_documents(ids: list[str], offset: int, limit: int):
    """Retrieve documents"""
    try:
        selected_files = load_client().documents.list(
            ids,
            offset,
            limit
        ).results
        if selected_files:
            for i, doc in enumerate(selected_files):
                with st.expander(
                    label=f"Document: {doc.title} ({doc.id})",
                    expanded=False
                ):
                    st.json(doc)

                    with st.popover(
                            label="Delete document",
                            icon="🗑️",
                        ):
                            delete_doc_btn = st.button( # pylint: disable=W0612
                                label="Confirm deletion",
                                key=f"delete_document_{i}",
                                on_click=delete_document,
                                args=(doc.id, )
                            )

            # Show a message if we've reached the end
            if len(selected_files) < limit:
                st.info("You've reached the end of the documents.")
        else:
            st.info("No documents found.")
    except R2RException as r2re:
        st.error(f"Error: {str(r2re)}")
    except Error as e:
        st.error(f"Error: {str(e)}")

def ingest_file(uploaded_file, metadata: dict):
    """Creating a temp file and ingesting it"""
    try:
        file_path = Path(st.session_state['files_dir']) / uploaded_file.name
        file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(file=file_path, mode="wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(text="Ingesting document...", show_time=True):
            if metadata:
                metadata = json.loads(metadata)

            ingest_resp = load_client().documents.create(
                file_path=str(file_path),
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
        st.error(f"Error: {str(e)}")
    finally:
        file_path.unlink()

def perform_webscrape(uploaded_url_file):
    """By providing a file with URLs we can scrape and ingest the data"""

    with st.status(
        label="Processing URLs...",
        expanded=True,
        state="running"
    ):
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
                            chunks_metadata = chunks[0].metadata
                            chunks_text = [chunk.page_content for chunk in chunks]
                            try:
                                chunks_ing_resp = load_client().documents.create(
                                    chunks=chunks_text,
                                    ingestion_mode='custom',
                                    metadata=chunks_metadata,
                                    run_with_orchestration=True
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

def export_docs_to_csv(filename: str, filetype: str, ingestion_status: str):
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

        out_path = Path(st.session_state['export_dir']) / f"{filename}.csv"
        load_client().documents.export(
            output_path=out_path,
            filters=filters,
            columns=columns,
            include_header=True
        )
        st.success("Successfully exported documents!")
    except R2RException as r2re:
        st.error(f"Error: {str(r2re)}")
    except Error as e:
        st.error(f"Error: {str(e)}")

def fetch_document_chunks(document_id: str, offset: int, limit: int):
    """Fetches all chunks related to a document"""
    try:
        chunks = load_client().documents.list_chunks(
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
        st.error(f"Error: {str(r2re)}")
    except Error as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__page__":
    st.title("📄 Document Management")

    t_list, t_chunks, t_file_ingest, t_webscrape, t_export_docs, t_download = st.tabs(
        [
            "List Documents",
            "List chunks",
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
            offset = st.number_input("Offset", min_value=0, value=0, step=10)
        with col2:
            limit = st.number_input("Limit", min_value=1, max_value=100, value=10, step=10)

        doc_ids = st.text_area(
            label="Document IDs",
            placeholder="document_id1\ndocument_id2\n...",
            height=100,
            value=None
        )

        if doc_ids:
            doc_ids = [doc.strip() for doc in doc_ids.split("\n")]

        if st.button("Fetch Documents", type="primary"):
            fetch_documents(doc_ids, offset, limit)

    with t_chunks:
        st.markdown("**List Chunks**")

        col1, col2 = st.columns([1, 2])
        with col1:
            offset = st.number_input("Offset", min_value=0, value=0, step=10, key="Chunks offset")
        with col2:
            limit = st.number_input("Limit", min_value=1, max_value=1000, value=10, step=10, key="Chunks limit")

        document_id_chunks = st.text_input(
            label="Document id",
            placeholder="Ex. document_id",
            value=None
        )

        if st.button("Fetch Chunks", type="primary"):
            if not document_id_chunks:
                st.error("Please provide a document id.")
            else:
                fetch_document_chunks(document_id_chunks, offset, limit)

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
            if not uploaded_file:
                st.error("Please upload a file.")
            else:
                ingest_file(uploaded_file, metadata)

    with t_webscrape:
        st.markdown("**Perform Web Scrape**")

        uploaded_url_file = st.file_uploader(
            label="Choose file containing URLs",
            type="csv",
            help="Supported formats: CSV"
        )

        if st.button("Ingest data from URLs", type="primary"):
            if not uploaded_url_file:
                st.error("Please upload a file containing URLs.")
            else:
                perform_webscrape(uploaded_url_file)

    with t_export_docs:
        st.markdown("**Export Documents**")

        files_csv_out = st.text_input(
            label='Name of output file (without extension)',
            placeholder="Ex. exported_docs"
        )

        filetype_col, ingestion_status_col = st.columns(2)

        with filetype_col:
            filetype_filter = st.selectbox(
                label="File type to filter on",
                options=["all", "csv", "txt", "pdf", "docx", "json"],
            )

        with ingestion_status_col:
            ingestion_status_filter = st.selectbox(
                label="Ingestion status to filter on",
                options=["all", "success", "embedding", "parsing"]
            )

        if st.button("Export Documents", type="primary"):
            if not files_csv_out:
                st.warning("Please enter a file name")
            else:
                export_docs_to_csv(
                    files_csv_out.strip(),
                    filetype_filter,
                    ingestion_status_filter
                )

    with t_download:
        st.markdown("**Download Documents**")

        download_out = st.text_input(
            label='Name of zip file to download (no extension)',
            placeholder="Ex. documents"
        )

        col1, col2 = st.columns(2)

        with col1:
            use_date_filter = st.checkbox("Filter by date range")

            if use_date_filter:
                start_date_filter = st.date_input(
                    label="Start date",
                    value=datetime(2025, 1, 1),
                    format="DD-MM-YYYY"
                )

                end_date_filter = st.date_input(
                    label="End date",
                    value=datetime.now(),
                    format="DD-MM-YYYY"
                )

        with col2:
            use_id_filter = st.checkbox("Filter by document IDs")

            if use_id_filter:
                document_ids_input = st.text_area(
                    label="Document IDs (one per line)",
                    placeholder="Enter document IDs, one per line",
                    height=125,
                    value=None
                )

        if st.button("Download Documents", type="primary"):
            if not download_out:
                st.error("Please provide a name for the ZIP file.")
            else:
                try:
                    document_ids = []
                    if use_id_filter:
                        document_ids = [
                            doc_id.strip()
                            for doc_id in document_ids_input.split('\n')
                            if doc_id.strip()
                        ]

                    START_DATE = None
                    END_DATE = None
                    if use_date_filter:
                        # Convert date to datetime at beginning and end of day
                        START_DATE = datetime.combine(start_date_filter, datetime.min.time())
                        END_DATE = datetime.combine(end_date_filter, datetime.max.time())

                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    out_path = Path(st.session_state['export_dir']) / f"{download_out}_{timestamp}.zip"
                    with st.spinner("Preparing documents for download..."):
                        load_client().documents.download_zip(
                            document_ids=document_ids,
                            start_date=START_DATE,
                            end_date=END_DATE,
                            output_path=out_path
                        )

                    st.success("Successfully downloaded documents!")
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