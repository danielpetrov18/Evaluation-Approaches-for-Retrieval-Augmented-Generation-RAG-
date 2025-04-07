"""
GUI support for interacting with documents.
There's also support for performing a web scrape.
There's additionally a tab to perform a simple web search to gather URLs and a small summary.
"""

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=C0301

import streamlit as st
from st_app import load_client
from backend.storage import (
    delete_all_documents,
    fetch_documents,
    fetch_document_chunks,
    ingest_file,
    perform_webscrape,
    export_docs_to_csv,
    export_chunks_to_csv,
    perform_websearch
)

if __name__ == "__page__":
    st.title("📄 Document Management")

    with st.sidebar:
        with st.popover(
            label="Delete all documents",
            help="Remove all documents from knowledge base",
            icon="🗑️"
        ):
            delete_all_docs_btn = st.button(
                label="Confirm deletion",
                key="delete_all_docs_btn",
                on_click=delete_all_documents,
                args=(load_client(), )
            )

    t_list, t_chunks, t_file_ingest, t_webscrape, t_export_docs, t_export_chunks, t_websearch = st.tabs(
        [
            "List Documents",
            "List chunks",
            "Ingest File",
            "Webscrape URLs",
            "Export Documents",
            "Export Chunks",
            "Web Search"
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
            type=["txt", "pdf", "docx", "csv", "md", "html"]
        )

        metadata = st.text_area(
            label="Metadata (JSON format)",
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
                help="If left on `all` it selects all possible documents"
            )

        with ingestion_status_col:
            ingestion_status_filter = st.selectbox(
                label="Ingestion status to filter on",
                options=["all", "success", "embedding", "parsing", "failed"],
                help="If left on `all` it selects all possible documents"
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

    with t_export_chunks:
        st.markdown("**Export Chunks**")

        chunks_csv_out = st.text_input(
            label='Name of output file (without extension)',
            placeholder="Ex. exported_chunks"
        )

        if st.button("Export Chunks", type="primary"):
            if not chunks_csv_out:
                st.warning("Please enter a file name")
            else:
                export_chunks_to_csv(
                    load_client(),
                    chunks_csv_out
                )

    with t_websearch:
        st.markdown("**Web Search**")

        with st.sidebar:
            new_api_key = st.text_input(
                label="API key",
                value=st.session_state['websearch_api_key'],
                type="password"
            )

            if st.button(label="Save API key", key="save_api_key_btn"):
                if not new_api_key:
                    st.error("Please enter an API key.")
                elif 'sk-' not in new_api_key:
                    st.error("Please enter a valid API key.")
                else:
                    st.session_state['websearch_api_key'] = new_api_key

        with st.expander("Instructions on how to use it", expanded=True, icon="📖"):
            st.markdown("""
            * First go to this website: [langsearch](https://langsearch.com/)
            * Create a free account and login
            * Get an API key that looks like this: `sk-****************`
            * Submit your API key in the input box on the left
            * Finally, submit a query and number of web pages
            * You will get a list of web pages that match your query each one containing:
                * Title
                * URL
                * Chunk of the summary       
            * You can use the links to create a csv file to then ingest     
            """)

        query = st.text_input(
            label="Enter query",
            key="query_input",
            placeholder="What is the capital of France?"
        )

        results_to_return = st.slider(
            label="Number of results to return",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="The number of web pages to be considered by the tool"
        )

        if st.button("Search", type="primary", key="websearch_btn"):
            if not query:
                st.error("Please enter a query.")
            elif st.session_state['websearch_api_key'] is None:
                st.error("Please enter an API key.")
            else:
                perform_websearch(load_client(), query, results_to_return)
