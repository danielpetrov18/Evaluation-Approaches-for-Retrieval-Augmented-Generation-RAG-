# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=E0401

from typing import List, Union

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from st_app import (
    r2r_client,
    ollama_client,
    ollama_options
)
from backend.storage import (
    delete_all_documents,
    fetch_documents,
    fetch_document_chunks,
    ingest_file,
    perform_webscrape,
    export_docs_to_csv,
    perform_websearch
)

if __name__ == "__page__":
    st.title("📄 Document Management")

    with st.sidebar:
        with st.popover(
            label="Delete all documents",
            help="Remove all documents from the knowledge base",
            icon="🗑️"
        ):
            delete_all_docs_btn = st.button(
                label="Confirm deletion",
                key="delete_all_docs_btn",
                on_click=delete_all_documents,
                args=(r2r_client(), )
            )

    t_list, t_chunks, t_file_ingest, t_webscrape, t_export_docs, t_websearch = st.tabs([
        "List Docs",
        "List Chunks",
        "Ingest File",
        "Webscrape",
        "Export Docs",
        "Web Search"
    ])

    with t_list:
        st.markdown("**List Documents**")

        col1, col2 = st.columns([1, 2])
        with col1:
            offset = st.number_input("Offset", min_value=0, value=0, step=10)
        with col2:
            limit = st.number_input("Limit", min_value=1, max_value=1000, value=10, step=10)

        doc_ids: str = st.text_area(
            label="Document IDs",
            placeholder="document_id1\ndocument_id2\n...",
            height=100,
            value=None
        )

        if doc_ids:
            doc_ids: List[str] = [doc.strip() for doc in doc_ids.strip().split("\n")]

        if st.button("Fetch Documents", type="primary", key="fetch_docs_btn"):
            fetch_documents(r2r_client(), doc_ids, offset, limit)

    with t_chunks:
        st.markdown("**List Chunks**")

        col1, col2 = st.columns([1, 2])
        with col1:
            offset: int = st.number_input(
                "Offset", 
                min_value=0,
                value=0,
                step=10,
                key="Chunks offset"
            )
        with col2:
            limit: int = st.number_input(
                "Limit",
                min_value=1,
                max_value=1000,
                value=10,
                step=10,
                key="Chunks limit"
            )

        document_id_chunks: str = st.text_input(
            label="Document id",
            placeholder="Ex. document_id",
            value=None
        )

        if st.button("Fetch Chunks", type="primary", key="fetch_chunks_btn"):
            document_id_chunks: str = document_id_chunks.strip()
            if not document_id_chunks:
                st.error("Please provide a document id.")
            else:
                fetch_document_chunks(r2r_client(), document_id_chunks, offset, limit)

    with t_file_ingest:
        st.markdown("**Ingest Document**")

        uploaded_file: Union[UploadedFile, None] = st.file_uploader(
            label="Choose a file to upload",
            type=["txt", "pdf", "docx", "csv", "md", "html", "json"]
        )

        metadata: str = st.text_area(
            label="Metadata (JSON format)",
            value="{}",
            help="Optional metadata in JSON format"
        )

        if st.button("Ingest Document", type="primary", key="ingest_doc_btn"):
            if not uploaded_file:
                st.error("Please upload a file.")
            else:
                ingest_file(r2r_client(), uploaded_file, metadata.strip())

    with t_webscrape:
        st.markdown("**Perform Web Scrape**")

        uploaded_url_file: Union[UploadedFile, None] = st.file_uploader(
            label="Choose file containing URLs",
            type="csv",
            help="Supported formats: CSV"
        )

        if st.button("Ingest data from URLs", type="primary", key="webscrape_btn"):
            if not uploaded_url_file:
                st.error("Please upload a file containing URLs.")
            else:
                perform_webscrape(r2r_client(), uploaded_url_file)

    with t_export_docs:
        st.markdown("**Export Documents**")

        files_csv_out: str = st.text_input(
            label='Name of output file (without extension)',
            placeholder="Ex. exported_docs"
        )

        ingestion_status_filter = st.selectbox(
            label="Ingestion status to filter on",
            options=["all", "success", "embedding", "parsing", "failed"],
            help="If left on `all` it selects all possible documents"
        )

        if st.button("Export Documents", type="primary", key="export_docs_btn"):
            if not files_csv_out:
                st.warning("Please enter a file name")
            else:
                export_docs_to_csv(
                    r2r_client(),
                    files_csv_out.strip(),
                    ingestion_status_filter
                )

    with t_websearch:
        st.markdown("**Web Search**")

        with st.sidebar:
            new_api_key: str = st.text_input(
                label="API key",
                value=st.session_state['websearch_api_key'],
                type="password"
            )

            if st.button(label="Save API key", key="save_api_key_btn"):
                if not new_api_key:
                    st.error("Please enter an API key.")
                elif 'sk-' not in new_api_key:
                    st.error("Please enter a valid API key.")
                elif new_api_key == st.session_state['websearch_api_key']:
                    st.error("Please enter a new API key.")
                else:
                    st.session_state['websearch_api_key'] = new_api_key.strip()
                    st.success("API key saved.")

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

        query: str = st.text_input(
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
            if not query.strip():
                st.error("Please enter a query.")
            elif st.session_state['websearch_api_key'] is None:
                st.error("Please enter an API key.")
            else:
                with st.spinner("Performing web search...", show_time=True):
                    result, urls = perform_websearch(
                        ollama_client(),
                        ollama_options(),
                        query,
                        results_to_return
                    )

                formatted_urls: str = ""
                for i, url in enumerate(urls, 0):
                    formatted_urls += f"{i}. [{url}]({url})\n"

                st.markdown(f"""### Response
{result}

### Relevant URLs:
{formatted_urls}
                """)
