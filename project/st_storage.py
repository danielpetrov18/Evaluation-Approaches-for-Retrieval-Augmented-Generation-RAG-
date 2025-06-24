# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=E0401

from typing import Union

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from st_app import KEY_FILE
from backend.storage import (
    delete_all_documents,
    fetch_documents,
    fetch_document_chunks,
    ingest_file,
    perform_webscrape,
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
                on_click=delete_all_documents
            )

        st.markdown("""
### About Document Storage in R2R

Documents in R2R are the core knowledge units used for retrieval and answering user queries. During ingestion, each document is broken down into **semantic chunks** and indexed for similarity search.

---

**What you can do here:**

- **List Docs**: View all documents in your current knowledge base.
- **List Chunks**: Inspect the individual content chunks and metadata per document.
- **Ingest File**: Upload supported files (`.txt`, `.pdf`, `.docx`, etc.) to be chunked and stored.
- **Web Search**: Use an LLM tool-call to fetch a response relative to query and URLs containing information (perfect for webscraping).
- **Webscrape**: Upload a csv file with URLs, scrape content from each, and ingest it as documents.

""")

    t_list, t_chunks, t_file_ingest, t_websearch, t_webscrape = st.tabs([
        "List Docs",
        "List Chunks",
        "Ingest File",
        "Web Search",
        "Webscrape"
    ])

    with t_list:
        if st.button("Fetch all documents", type="primary", key="fetch_docs_btn"):
            fetch_documents()

    with t_chunks:
        document_id_chunks: str = st.text_input(
            label="Enter document id and retrieve corresponding chunks",
            placeholder="Ex. document_id",
            value=None
        )

        if st.button("Fetch Chunks", type="primary", key="fetch_chunks_btn"):
            if not document_id_chunks:
                st.error("Please provide a document id.")
            else:
                fetch_document_chunks(document_id_chunks.strip())

    with t_file_ingest:
        uploaded_file: Union[UploadedFile, None] = st.file_uploader(
            label="Choose a file to upload",
            type=["txt", "pdf", "docx", "csv", "md", "html", "json"]
        )

        if st.button("Ingest Document", type="primary", key="ingest_doc_btn"):
            if not uploaded_file:
                st.error("Please upload a file.")
            else:
                ingest_file(uploaded_file)

    with t_websearch:
        with st.expander("Instructions on how to use it", expanded=True, icon="📖"):
            st.markdown(f"""
            * First go to this website: [langsearch](https://langsearch.com/)
            * Create a free account and login
            * Get an API key that looks like this: `sk-****************`
            * Enter your API key in the field below. It will store it under `{KEY_FILE}`.
            * Finally, submit a query and number of web pages
            * You will get a response and a list of web pages that match your query       
            * You can use the links to create a csv file to then perform a webscrape
            """)

        provided_api_key: str = st.text_input(
            label="Enter your API key",
            key="api_key_input",
            value=st.session_state['websearch_api_key']
        )

        if st.button("Set API key", type="primary", key="set_api_key_btn"):
            if not provided_api_key:
                st.error("Please enter an API key.")
            else:
                if not provided_api_key.startswith("sk-"):
                    st.error("Please enter a valid API key.")
                elif provided_api_key == st.session_state['websearch_api_key']:
                    st.error("Please enter a different API key.")
                else:
                    st.session_state['websearch_api_key'] = provided_api_key.strip()
                    # Persist the key
                    KEY_FILE.write_text(provided_api_key.strip(), encoding="utf-8")
                    st.success("API key updated and persisted.")

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
            if not st.session_state['websearch_api_key']:
                st.error("Please enter an API key.")
            else:
                if not query:
                    st.error("Please enter a query.")
                else:
                    with st.spinner("Performing web search...", show_time=True):
                        result, urls = perform_websearch(query.strip(), results_to_return)

                    formatted_urls: str = ""
                    for i, url in enumerate(urls, 1):
                        formatted_urls += f"{i}. [{url}]({url})\n"

                    st.markdown(f"""### Response
{result}

### Relevant URLs:
{formatted_urls}
                """)

    with t_webscrape:
        uploaded_url_file: Union[UploadedFile, None] = st.file_uploader(
            label="Choose file containing URLs",
            type="csv",
            help="Supported formats: CSV"
        )

        if st.button("Ingest data from URLs", type="primary", key="webscrape_btn"):
            if not uploaded_url_file:
                st.error("Please upload a file containing URLs.")
            else:
                perform_webscrape(uploaded_url_file)
