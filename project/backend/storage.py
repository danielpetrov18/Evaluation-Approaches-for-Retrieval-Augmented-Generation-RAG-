"""Backend functionality for interacting with documents."""

# pylint: disable=E0401
# pylint: disable=W0612
# pylint: disable=W0718
# pylint: disable=W0719
# pylint: disable=R0914
# pylint: disable=R1732

import os
import time
import asyncio
import tempfile
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Union

import json
import requests
import pandas as pd
from ollama import Options, Client
from r2r import R2RException, R2RClient

import streamlit as st
from streamlit.errors import Error
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain.docstore.document import Document
from langchain_community.document_loaders import AsyncHtmlLoader

@st.cache_resource
def ascrapper(urls: List[str]):
    """Load object to retrieve data from the internet"""
    return AsyncHtmlLoader(
        web_path=urls,
        default_parser="lxml"
    )

@st.cache_resource
def ollama_tools() -> List[Dict[str, Union[str, Dict]]]:
    """Single tool available to be called by Ollama"""
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

def delete_all_documents(client: R2RClient):
    """Delete all present documents"""
    try:
        doc_ids = [doc.id for doc in client.documents.list().results]
        for doc_id in doc_ids:
            client.documents.delete(doc_id)
        st.success("Successfully deleted all documents")
    except R2RException as r2re:
        st.error(f"Error when deleting document: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def fetch_documents(
    client: R2RClient,
    ids: List[str],
    offset: int,
    limit: int
):
    """Retrieve documents. For each document there's a delete and update metadata buttons."""
    try:
        selected_files = client.documents.list(ids, offset, limit).results
        if selected_files:
            for i, doc in enumerate(selected_files):
                with st.expander(label=f"{i + 1}: {doc.title}", expanded=False):
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
        st.error(f"Error when fetching documents: {r2re.message}")
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
        st.error(f"Error when deleting document: {r2re.message}")
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
        st.error(f"Error: {r2re.message}")
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
        st.error(f"Error when fetching chunks for document: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def ingest_file(client: R2RClient, file: UploadedFile, metadata: dict):
    """
    This method takes a file in binary format. Saves it in the tmp folder.
    Thereafter, the text is extracted and split. Finally, embeded vectors are created
    and stored in the database.
    """
    # Do it outside because of the finally clause
    temp_filepath = os.path.join(tempfile.gettempdir(), file.name)
    try:
        # Make sure file doesn't already exist.
        for doc in client.documents.list().results:
            if doc.title == file.name:
                st.error("File already exists!")
                return

        with open(file=temp_filepath, mode="wb") as temp_file:
            temp_file.write(file.getbuffer())

        if not os.path.exists(temp_filepath) or os.path.getsize(temp_filepath) == 0:
            st.error("Failed to save file or file is empty.")
            return

        with st.spinner(text="Ingesting document...", show_time=True):
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            ingest_resp = client.documents.create(
                file_path=temp_filepath,
                metadata=metadata,
                ingestion_config=st.session_state['ingestion_config'],
                run_with_orchestration=True
            ).results
            st.success(ingest_resp.message)
    except json.JSONDecodeError as jde:
        st.error(f"Error: {str(jde)}")
    except R2RException as r2re:
        st.error(f"Error: {r2re.message}")
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
                    # Save file temporarily
                    temp_filepath = os.path.join(tempfile.gettempdir(), file.name)
                    with open(file=temp_filepath, mode="wb") as temp_file:
                        temp_file.write(document.page_content)
                        temp_file.flush()

                    try:
                        chunks_ing_resp = client.documents.create(
                            file_path=temp_file,
                            ingestion_mode='fast',
                            metadata=document.metadata,
                            run_with_orchestration=True
                        ).results
                        st.success(f"{document.metadata['source']}: {chunks_ing_resp.message}")
                    except R2RException as r2re:
                        st.error(f"Error {document.metadata['source']}: {r2re.message}")
                    finally:
                        if os.path.exists(temp_filepath):
                            os.remove(temp_filepath)

                    time.sleep(5) # Wait for ingestion
                st.info("Completed URL ingestion process")
            else:
                st.error("No valid URLs found in file")
        except R2RException as r2re:
            st.error(f"Error: {r2re.message}")
        except ValueError as ve:
            st.error(f"Error: {str(ve)}")
        except Error as e:
            st.error(f"Unexpected streamlit error: {str(e)}")
        except Exception as exc:
            st.error(f"Unexpected error: {str(exc)}")

def export_docs_to_csv(client: R2RClient, filename: str, ingestion_status: str):
    """Exports all available documents to a csv file."""
    try:
        filters = {}

        if ingestion_status != "all":
            filters["ingestion_status"] = ingestion_status

        columns = [
            "id",
            "type",
            "title",
            "ingestion_status",
            "created_at",
            "updated_at"
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
        st.error(f"Error: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def perform_websearch(
    ollama_client: Client,
    options: Options,
    query: str,
    results_to_return: int
) -> tuple[str, List[str]]:
    """
    Uses the following API https://langsearch.com/ to perform a web search.
    Then we can receive the results and use them as context for generating data.
    However, this doesn't use R2R, but simple Ollama with a tool call.
    """
    try:
        # Call the model with the properly formatted tools
        response = ollama_client.chat(
            model=st.session_state['chat_model'],
            options=options,
            messages=[
                {
                    'role': 'user',
                    'content': query
                }
            ],
            tools=ollama_tools()  # Pass the structured tools object
        )

        if "message" in response and "tool_calls" in response["message"]:
            for tool_call in response["message"]["tool_calls"]:
                if tool_call["function"]["name"] == "langsearch_websearch_tool":
                    search_results, urls = _langsearch_websearch_tool(
                        query=query, count=results_to_return
                    )

                    # Continue the conversation with the tool results
                    final_response = ollama_client.chat(
                        model=st.session_state['chat_model'],
                        options=options,
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
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def _langsearch_websearch_tool(query: str, count: int) -> tuple[str, List[str]]:
    """Performs a web search and returns the data in a formated way."""

    url = "https://api.langsearch.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {st.session_state['websearch_api_key']}",
        "Content-Type": "application/json"
    }
    data = {
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": count
    }

    response = requests.post(
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

            webpages = json_response["data"]["webPages"]["value"]
            if not webpages:
                return "No relevant results found.", []

            formatted_results = ""
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

def _fetch_data_from_urls(scrape_urls: List[str]) -> List[Document]:
    """Fetches data from the provided URLs asynchronously."""
    scraper = ascrapper(scrape_urls)
    web_documents = _run_async_function(scraper.aload())
    return web_documents

def _extract_urls(file: UploadedFile) -> List[str]:
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

def _remove_duplicate_urls(urls: List[str]) -> List[str]:
    return list(set(urls))
