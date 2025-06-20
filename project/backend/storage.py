# pylint: disable=C0114
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=C0303
# pylint: disable=E0401
# pylint: disable=R0914
# pylint: disable=R1732
# pylint: disable=W0612
# pylint: disable=W0718
# pylint: disable=W0719

import os
import time
import asyncio
import hashlib
import tempfile
import mimetypes
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict, Union, Any

import requests
import pandas as pd
from ollama import Options, Client

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain.docstore.document import Document
from langchain_community.document_loaders import AsyncHtmlLoader

@st.cache_resource
def ollama_client():
    return Client(host=st.session_state['ollama_api_base'])

@st.cache_resource
def ollama_options():
    return Options(
        temperature=st.session_state['temperature'],
        top_p=st.session_state['top_p'],
        top_k=st.session_state['top_k'],
        num_ctx=st.session_state["context_window_size"],
        format="json", # This should be json to enforce proper output if required
    )

@st.cache_resource
def ollama_tools() -> List[Dict[str, Union[str, Dict]]]:
    """
    Custom tool, which can be used by `Ollama`.
    After submitting a request to the `ollama` client, the response will contain the tool calls.
    From there one can call the `langsearch_websearch_tool`
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "langsearch_websearch_tool",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

@st.cache_resource
def ascrapper(urls: List[str]):
    return AsyncHtmlLoader(
        web_path=urls,
        default_parser="lxml"
    )

def _retrieve_documents():
    response: requests.Response = requests.get(
        url="http://r2r:7272/v3/documents",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        params={
            "limit": 1000 # Max documents
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Failed to fetch documents: {response.status_code} - {response.text}")
        return []

    documents: List[Dict] = response.json()['results']
    return documents

def delete_all_documents():
    doc_ids: List[str] = [doc['id'] for doc in _retrieve_documents()]
    for doc_id in doc_ids:
        delete_document(doc_id)
    st.info("Successfully deleted all documents")

def fetch_documents():
    documents: List[Dict] = _retrieve_documents()
    if not documents:
        st.info("No documents found.")
        return

    for i, doc in enumerate(documents, 1):
        with st.expander(label=f"{i}: {doc['title']}", expanded=False):
            st.markdown(f"""ID: `{doc['id']}`  
Title: `{doc['title']}`  
Version: `{doc['version']}`   
Size in bytes: `{doc['size_in_bytes']:,}`  
Ingestion status: `{doc['ingestion_status']}`  
Created at: `{datetime.fromisoformat(doc['created_at'])}`   
""")

            with st.popover(label="Delete document", icon="🗑️"):
                delete_doc_btn = st.button(
                    label="Confirm deletion",
                    key=f"delete_document_{i}",
                    on_click=delete_document,
                    args=(doc['id'], )
                )

    st.info("You've reached the end of the documents.")

def delete_document(document_id: str):
    response: requests.Response = requests.delete(
        url=f"http://r2r:7272/v3/documents/{document_id}",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        timeout=5
    )
    
    if response.status_code != 200:
        st.error(f"Failed to delete document: {response.status_code} - {response.text}")
        return
    
    st.success(f"Successfully deleted document: {document_id}")

def fetch_document_chunks(document_id: str):
    response: requests.Response = requests.get(
        url=f"http://r2r:7272/v3/documents/{document_id}/chunks",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        params={
            "limit": 1000 # Max chunks
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Failed to fetch chunks: {response.status_code} - {response.text}")
        return
    
    chunks: List[Dict] = response.json()['results']
    
    if not chunks:
        st.info("No chunks found.")
        return

    for i, chunk in enumerate(chunks, 1):
        st.markdown("### Text: \n", unsafe_allow_html=True)
        st.markdown(chunk['text'])
        st.markdown("### Metadata: \n", unsafe_allow_html=True)
        for k, v in chunk['metadata'].items():
            st.markdown(f"* **{k.upper()}**: `{v}`")

def ingest_file(file: UploadedFile):
    # First save the file into the tmp folder
    # R2R receives a filepath so we need to have it
    # Do it outside because of the finally clause
    temp_filepath: str = os.path.join(tempfile.gettempdir(), file.name)
    try:
        # Step 1: Check if file was already ingested
        documents: List[Dict] = _retrieve_documents()     
        for doc in documents:
            if doc['title'] == file.name:
                st.error("File already exists!")
                return

        # Step 2: Save uploaded file to disk
        with open(file=temp_filepath, mode="wb") as temp_file:
            temp_file.write(file.getbuffer())
            temp_file.flush()

        if not os.path.exists(temp_filepath) or os.path.getsize(temp_filepath) == 0:
            st.error("Failed to save file or file is empty!")
            return

        mime_type, _ = mimetypes.guess_type(temp_filepath)
        if mime_type is None:
            mime_type = "application/octet-stream"

        # Step 3: Ingest file
        with open(temp_filepath, "rb") as f:
            with st.spinner(text="Ingesting document...", show_time=True):
                response: requests.Response = requests.post(
                    url="http://r2r:7272/v3/documents",
                    headers={
                        "Authorization": f"Bearer {st.session_state['bearer_token']}"
                    },
                    files={
                        "file": (file.name, f, mime_type)
                    },
                    json={
                        "ingestion_mode": "custom",
                        "ingestion_config": st.session_state['ingestion_config'],
                    },
                    timeout=3600 # 1 hour timeout for ingestion 
                )
                
                if response.status_code != 202:
                    st.error(f"Failed to ingest document: {response.status_code} - {response.text}")
                    return
                
                st.success(response.json()['results']['message'])
    finally:
        # Remove temporary file after ingestion
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

def perform_websearch(query: str, results_to_return: int) -> tuple[str, List[str]]:
    """
    Uses the following API https://langsearch.com/ to perform a web search.
    Then we can receive the results and use them as context for generating data.
    However, this doesn't use R2R, but simple Ollama with a tool call.
    """
    # Call the model with the properly formatted tools
    response: Dict[str, Any] = ollama_client().chat(
        model=st.session_state['chat_model'],
        options=ollama_options(),
        messages=[
            {
                'role': 'user',
                'content': query
            }
        ],
        tools=ollama_tools()
    )

    for tool_call in response["message"]["tool_calls"]: # Go over the tool calls
        if tool_call["function"]["name"] == "langsearch_websearch_tool":
            search_results, urls = _langsearch_websearch_tool(
                query=query, count=results_to_return
            )

            # Continue the conversation with the tool results
            final_response: Dict[str, Any] = ollama_client().chat(
                model=st.session_state['chat_model'],
                options=ollama_options(),
                messages=[
                    {
                        'role': 'user',
                        'content': query
                    },
                    {
                        'role': 'assistant',
                        'content': response["message"]["content"],
                        'tool_calls': response['message']["tool_calls"]
                    },
                    {
                        'role': 'tool',
                        'name': tool_call["function"]["name"],
                        'content': search_results
                    }
                ]
            )
            return final_response["message"]["content"], urls

    return "Nothing found", []

def _langsearch_websearch_tool(query: str, count: int) -> tuple[str, List[str]]:
    url: str = "https://api.langsearch.com/v1/web-search"
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {st.session_state['websearch_api_key']}",
        "Content-Type": "application/json"
    }
    data: Dict[str, Any] = {
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": count
    }

    response: requests.Response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=10
    )

    if response.status_code == 200:
        json_response = response.json()

        try:
            if json_response["code"] != 200 or not json_response["data"]:
                return f"Search API request failed, reason: {response.msg or 'Unknown error'}", []

            webpages: Any = json_response["data"]["webPages"]["value"]
            if not webpages:
                return "No relevant results found.", []

            formatted_results: str = ""
            urls = []
            for idx, page in enumerate(webpages, start=1):
                if len(page['summary']) > 1000:  # Limit content length
                    page['summary'] = page['summary'][:1000] + "..."

                formatted_results += (
                    f"Citation: {idx}\n"
                    f"Title: {page['name']}\n"
                    f"URL: {page['url']}\n"
                    f"Content: {page['summary']}\n"
                )
                urls.append(page['url'])
            return formatted_results.strip(), urls
        except Exception as e:
            return f"Search API request failed, reason: Failed to parse search results {str(e)}", []
    else:
        return f"Search API request failed, ({response.status_code}: {response.text})", []

def perform_webscrape(file: UploadedFile):
    with st.status(
        label="Processing URLs...",
        expanded=True,
        state="running"
    ):
        try:
            urls: List[str] = _extract_urls(file)
            if len(urls) == 0:
                st.error("No valid URLs found in file")
                return

            st.write('Extracted URLs...')

            documents: List[Document] = _fetch_data_from_urls(urls)
            st.write('Fetched data...')

            existing_sources: List[str] = [
                doc['metadata'].get('source', 'unknown')
                for doc in _retrieve_documents()
            ]

            for document in documents:
                url: str = document.metadata.get('source', 'unknown')
                source_name: str = _safe_filename_from_url(url)
                if source_name in existing_sources:
                    st.warning(f"Document '{source_name}' already exists. Skipping.")
                    continue

                # Write scraped content to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                    temp_file.write(document.page_content.encode("utf-8"))
                    temp_file.flush()
                    temp_path = temp_file.name

                mime_type, _ = mimetypes.guess_type(temp_path)
                if mime_type is None:
                    mime_type = "text/plain"

                with open(temp_path, "rb") as f:
                    ingestion_resp = requests.post(
                        url="http://r2r:7272/v3/documents",
                        headers={
                            "Authorization": f"Bearer {st.session_state['bearer_token']}"
                        },
                        files={
                            "file": (f"{source_name}.txt", f, mime_type)
                        },
                        json={
                            "ingestion_mode": "custom",
                            "ingestion_config": st.session_state['ingestion_config'],
                            "metadata": document.metadata
                        },
                        timeout=3600
                    )

                os.remove(temp_path)
                time.sleep(5) # Wait for ingestion to complete

                if ingestion_resp.status_code == 202:
                    st.success(f"✅ Ingested: {source_name}")
                else:
                    st.error(f"❌ Failed to ingest {source_name}: {ingestion_resp.status_code}")
                    st.error(ingestion_resp.text)

            st.info("🎉 Web scraping and ingestion complete.")
        except ValueError as ve:
            st.error(f"Error: {str(ve)}")

def _extract_urls(file: UploadedFile) -> List[str]:
    dataframe = pd.read_csv(
        filepath_or_buffer=file,
        usecols=[0],
        header=None
    )

    if dataframe.empty:
        raise ValueError("CSV file is empty!")

    extracted_urls: List[str] = dataframe.iloc[0:, 0].dropna().astype(str).str.strip().tolist()
    extracted_urls: List[str] = _remove_duplicate_urls(extracted_urls)
    return extracted_urls

def _remove_duplicate_urls(urls: List[str]) -> List[str]:
    return list(set(urls))

def _fetch_data_from_urls(scrape_urls: List[str]) -> List[Document]:
    scraper = ascrapper(scrape_urls)
    web_documents: List[Document] = _run_async_function(scraper.aload())
    return web_documents

def _run_async_function(coroutine):
    return asyncio.run(coroutine)

def _safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.replace(".", "-")
    path = parsed.path.strip("/").replace("/", "-")
    if not path:
        path = "index"
    filename = f"{netloc}-{path}"
    # Optional: hash query string to avoid long names
    if parsed.query:
        hash_part = hashlib.md5(parsed.query.encode()).hexdigest()[:6]
        filename += f"-{hash_part}"
    return filename + ".txt"
