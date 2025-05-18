# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=E0401
# pylint: disable=R0914
# pylint: disable=R1732
# pylint: disable=W0612
# pylint: disable=W0718
# pylint: disable=W0719

import os
import time
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Union, Any

import json
import requests
import pandas as pd
from ollama import Options, Client
from r2r import R2RException, R2RClient

import streamlit as st
from streamlit.errors import Error
from streamlit.runtime.uploaded_file_manager import UploadedFile

from shared.abstractions.document import UUID, DocumentResponse
from shared.api.models.management.responses import ChunkResponse
from shared.api.models.ingestion.responses import IngestionResponse

from langchain.docstore.document import Document
from langchain_community.document_loaders import AsyncHtmlLoader

@st.cache_resource
def ascrapper(urls: List[str]):
    """
    This object is used to retrieve data from the internet using the specified URLs.
    It does so asynchronously. You can experiment with other parsers.
    """
    return AsyncHtmlLoader(
        web_path=urls,
        default_parser="lxml"
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

def delete_all_documents(client: R2RClient):
    """Delete all present documents"""
    try:
        doc_ids: List[UUID] = [doc.id for doc in client.documents.list().results]
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
    """Retrieve documents. For each document there's a delete button."""
    try:
        selected_files: List[DocumentResponse] = client.documents.list(ids, offset, limit).results
        if selected_files:
            for i, doc in enumerate(selected_files):
                with st.expander(label=f"{i + 1}: {doc.title}", expanded=False):
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

def fetch_document_chunks(client: R2RClient, document_id: str, offset: int, limit: int):
    """Fetches all chunks related to a document"""
    try:
        chunks: List[ChunkResponse] = client.documents.list_chunks(
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
    This method takes a file in binary format and saves it in the tmp folder.
    Thereafter, the text is extracted and split.
    Finally, embeddings are generated and stored in the database with additional metadata.
    """

    temp_filepath: str = os.path.join(tempfile.gettempdir(), file.name) # Do it outside because of the finally clause
    try:
        # Make sure file doesn't already exist.
        for doc in client.documents.list().results:
            if doc.title == file.name:
                st.error("File already exists!")
                return

        # Save content to temp file
        with open(file=temp_filepath, mode="wb") as temp_file:
            temp_file.write(file.getbuffer())
            temp_file.flush()

        if not os.path.exists(temp_filepath) or os.path.getsize(temp_filepath) == 0:
            st.error("Failed to save file or file is empty.")
            return

        with st.spinner(text="Ingesting document...", show_time=True):
            if isinstance(metadata, str):
                metadata: Any = json.loads(metadata)

            ingest_resp: IngestionResponse = client.documents.create(
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
            urls: List[str] = _extract_urls(file)
            if len(urls) == 0:
                st.error("No valid URLs found in file")
                return

            st.write('Extracted URLs...')

            documents: List[Document] = _fetch_data_from_urls(urls)
            st.write('Fetched data...')

            # Make sure there's no document with that specific source
            sources: List[DocumentResponse] = [
                doc.metadata.get('source', 'unknown')
                for doc in client.documents.list().results
            ]

            for document in documents:
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=True, suffix=".txt") as temp_file:
                    temp_file.write(document.page_content.encode('utf-8'))
                    temp_file.flush()

                    try:
                        # Make sure there's no document with that specific source
                        if document.metadata['source'] in sources:
                            st.error(
                                f"Document {document.metadata['source']} already exists"
                            )
                            continue

                        # No summary will be generated
                        # If you want a summary switch to custom mode
                        chunks_ing_resp: IngestionResponse = client.documents.create(
                            file_path=temp_file.name,
                            ingestion_mode='fast',
                            metadata=document.metadata,
                            run_with_orchestration=True
                        ).results
                        st.success(f"{document.metadata['source']}: {chunks_ing_resp.message}")
                        time.sleep(5) # Wait for ingestion to complete
                    except R2RException as r2re:
                        st.error(f"Error {document.metadata['source']}: {r2re.message}")

            st.info("Completed URL ingestion process")
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
        filters: Dict = {}

        if ingestion_status != "all":
            filters["ingestion_status"] = ingestion_status

        columns: List[str] = [
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
        response: Dict[str, Any] = ollama_client.chat(
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
                    final_response: Dict[str, Any] = ollama_client.chat(
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
        return "Nothing found", []
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")
        return "Nothing found", []

def _langsearch_websearch_tool(query: str, count: int) -> tuple[str, List[str]]:
    """Performs a web search and returns the data in a formated way."""

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

def _fetch_data_from_urls(scrape_urls: List[str]) -> List[Document]:
    scraper = ascrapper(scrape_urls)
    web_documents = _run_async_function(scraper.aload())
    return web_documents

def _extract_urls(file: UploadedFile) -> List[str]:
    if file is None:
        raise FileNotFoundError("File not found")

    filename: str = str(file.name)
    file_extension: str = Path(filename).suffix.lower()

    if file_extension != ".csv":
        raise ValueError(f"Unsupported file type: {file_extension}")

    dataframe = pd.read_csv(
        filepath_or_buffer=file,
        usecols=[0],
        header=None
    )

    if dataframe.empty:
        raise ValueError("CSV file is empty")

    extracted_urls: List[str] = dataframe.iloc[1:, 0].dropna().astype(str).str.strip().tolist()
    extracted_urls: List[str] = _remove_duplicate_urls(extracted_urls)
    return extracted_urls

def _run_async_function(coroutine):
    return asyncio.run(coroutine)

def _remove_duplicate_urls(urls: List[str]) -> List[str]:
    return list(set(urls))
