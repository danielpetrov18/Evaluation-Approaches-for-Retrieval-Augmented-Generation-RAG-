"""GUI support for interacting with documents."""

# pylint: disable=E0401
# pylint: disable=C0103

import datetime
import streamlit as st
from st_app import load_client
from utility.r2r.documents import (
    fetch_documents,
    fetch_document_chunks,
    ingest_file,
    perform_webscrape,
    export_docs_to_csv,
    download_documents
)

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
            fetch_documents(load_client(), doc_ids, offset, limit)

    with t_chunks:
        st.markdown("**List Chunks**")

        col1, col2 = st.columns([1, 2])
        with col1:
            offset = st.number_input(
                "Offset", 
                min_value=0,
                value=0,
                step=10,
                key="Chunks offset"
            )
        with col2:
            limit = st.number_input(
                "Limit",
                min_value=1,
                max_value=1000,
                value=10,
                step=10,
                key="Chunks limit"
            )

        document_id_chunks = st.text_input(
            label="Document id",
            placeholder="Ex. document_id",
            value=None
        )

        if st.button("Fetch Chunks", type="primary", key="fetch_chunks_btn"):
            if not document_id_chunks:
                st.error("Please provide a document id.")
            else:
                fetch_document_chunks(load_client(), document_id_chunks, offset, limit)

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

        if st.button("Ingest Document", type="primary", key="ingest_doc_btn"):
            if not uploaded_file:
                st.error("Please upload a file.")
            else:
                ingest_file(load_client(), uploaded_file, metadata)

    with t_webscrape:
        st.markdown("**Perform Web Scrape**")

        uploaded_url_file = st.file_uploader(
            label="Choose file containing URLs",
            type="csv",
            help="Supported formats: CSV"
        )

        if st.button("Ingest data from URLs", type="primary", key="webscrape_btn"):
            if not uploaded_url_file:
                st.error("Please upload a file containing URLs.")
            else:
                perform_webscrape(load_client(), uploaded_url_file)

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
                options=["all", "success", "embedding", "parsing", "failed"]
            )

        if st.button("Export Documents", type="primary"):
            if not files_csv_out:
                st.warning("Please enter a file name")
            else:
                export_docs_to_csv(
                    load_client(),
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
                    value=datetime.datetime(2025, 1, 1),
                    format="DD-MM-YYYY"
                )

                end_date_filter = st.date_input(
                    label="End date",
                    value=datetime.datetime.now(),
                    format="DD-MM-YYYY"
                )
            else:
                start_date_filter = None
                end_date_filter = None

        with col2:
            use_id_filter = st.checkbox("Filter by document IDs")

            if use_id_filter:
                document_ids_input = st.text_area(
                    label="Document IDs (one per line)",
                    placeholder="Enter document IDs, one per line",
                    height=125,
                    value=None
                )
            else:
                document_ids_input = None

        if st.button("Download Documents", type="primary"):
            if not download_out:
                st.error("Please provide a name for the ZIP file.")
            else:
                download_documents(
                    load_client(),
                    download_out,
                    document_ids_input,
                    start_date_filter,
                    end_date_filter
                )
