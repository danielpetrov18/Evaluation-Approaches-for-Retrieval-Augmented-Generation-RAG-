# pylint: disable=C0114
# pylint: disable=C0116
# pylint: disable=C0301

import os
from typing import List

import requests
import streamlit as st
from streamlit.navigation.page import StreamlitPage

def get_pages() -> List[StreamlitPage]:
    return [
        st.Page(
            page="st_chat.py",
            title="Chatbot",
            icon=":material/chat:",
            url_path="chat",
            default=True
        ),
        st.Page(
            page="st_storage.py",
            title="Documents",
            url_path="documents",
            icon=":material/docs:"
        ),
        st.Page(
            page="st_conversation.py",
            title="Conversations",
            url_path="conversations",
            icon=":material/forum:"
        ),
        st.Page(
            page="st_prompt.py",
            title="Prompts",
            url_path="prompts",
            icon=":material/notes:"
        ),
        st.Page(
            page="st_index.py",
            title="Indices",
            url_path="index",
            icon=":material/description:"
        )
    ]

if __name__ == "__main__":
    pages: List[StreamlitPage] = get_pages()

    # Register pages. Creates the navigation menu for the application.
    # This page is an entrypoint and as such serves as a page router.
    page: StreamlitPage = st.navigation(pages)

    # ====== TWEAK VALUES BELOW TO ACHIEVE BEST PERFORMANCE ======

    if "top_k" not in st.session_state:
        st.session_state['top_k'] = int(os.getenv("TOP_K"))

    if "top_p" not in st.session_state:
        st.session_state['top_p'] = float(os.getenv("TOP_P"))

    if "max_tokens" not in st.session_state:
        st.session_state['max_tokens'] = int(os.getenv("MAX_TOKENS"))

    if "chunk_size" not in st.session_state:
        st.session_state['chunk_size'] = int(os.getenv("CHUNK_SIZE"))

    if "chunk_overlap" not in st.session_state:
        st.session_state['chunk_overlap'] = int(os.getenv("CHUNK_OVERLAP"))

    if "temperature" not in st.session_state:
        st.session_state['temperature'] = float(os.getenv("TEMPERATURE"))

    if "chat_model" not in st.session_state:
        st.session_state["chat_model"] = os.getenv("CHAT_MODEL")

    if "embedding_model" not in st.session_state:
        st.session_state["embedding_model"] = os.getenv("EMBEDDING_MODEL")

    # ====== TWEAK VALUES ABOVE TO ACHIEVE BEST PERFORMANCE ======

    if "conversation_id" not in st.session_state:
        st.session_state['conversation_id'] = None

    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    # The id of the last message in a given conversation
    if "parent_id" not in st.session_state:
        st.session_state["parent_id"] = None

    if "context_window_size" not in st.session_state:
        st.session_state["context_window_size"] = int(os.getenv("LLM_CONTEXT_WINDOW_TOKENS"))

    if 'ingestion_config' not in st.session_state:
        response: requests.Response = requests.get(
            url="http://r2r:7272/v3/system/settings",
            timeout=5
        )

        if response.status_code != 200:
            st.error(f"Failed to fetch system settings: {response.status_code} - {response.text}")
        else:
            st.session_state['ingestion_config'] = response.json()['results']['config']['ingestion']

            # Since the config is a snapshot not an actual instance of the config
            new_ingestion_config = st.session_state['ingestion_config']

            # During ingestion, we need to extract the text from the documents.
            # Then they are chunked. If unstructured cannot handle it, a fallback is used.
            # RecursiveCharacterTextSplitter is the fallback.
            #
            # https://docs.unstructured.io/api-reference/partition/chunking
            new_ingestion_config['extra_fields']['max_characters'] = st.session_state['chunk_size']
            new_ingestion_config['extra_fields']['overlap'] = st.session_state['chunk_overlap']
            new_ingestion_config['extra_fields']['new_after_n_chars'] = (
                new_ingestion_config['extra_fields']['max_characters']
            )
            new_ingestion_config['extra_fields']['combine_text_under_n_chars'] = int(
                int(new_ingestion_config['extra_fields']['max_characters']) / 2
            )

            # This will be the same ingestion config, however we could overwrite it
            # using environment variables from `env/rag.env`.
            st.session_state['ingestion_config'] = new_ingestion_config

    # Login values are default ones. Can be modified in the config file.
    if "bearer_token" not in st.session_state:
        response: requests.Response = requests.post(
            url="http://r2r:7272/v3/users/login",
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            },
            data={
                "username": "admin@example.com",
                "password": "change_me_immediately"
            },
            timeout=5
        )

        if response.status_code != 200:
            st.error(f"Failed to fetch system settings: {response.status_code} - {response.text}")
            st.write("KURWA")
        else:
            st.session_state['bearer_token'] = response.json()['results']['access_token']['token']

    # Default prompt name that is used by R2R when interacting with /rag endpoint
    # You can specify a custom name in the application itself
    if 'selected_prompt' not in st.session_state:
        st.session_state['selected_prompt'] = "rag"

    # The actual template of the prompt
    if 'prompt_template' not in st.session_state:
        response: requests.Response = requests.post(
            url=f"http://r2r:7272/v3/prompts/{st.session_state['selected_prompt']}",
            headers={
                "Authorization": f"Bearer {st.session_state['bearer_token']}"
            },
            timeout=5
        )

        if response.status_code != 200:
            st.error(f"Failed to fetch system settings: {response.status_code} - {response.text}")
        else:
            st.session_state['prompt_template'] = response.json()['results']['template']

    # It's part of a tool call, that can fetch data from the internet.
    if 'websearch_api_key' not in st.session_state:
        # Bad practice to hardcode, I know
        st.session_state['websearch_api_key'] = "sk-47eac60b82c844fe9dbd4f7e98de4951"

    if "ollama_api_base" not in st.session_state:
        st.session_state['ollama_api_base'] = os.getenv("OLLAMA_API_BASE")

    # Run selected page
    page.run()
