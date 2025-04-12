"""
This module holds all the available pages of the application.
Every resource defined in the main page can be accessed by all pages.
For that reason I use caching mechanism for faster page loading.
"""

import os
import typing as t
from ollama import (
    Client,
    Options
)
from r2r import R2RClient
import streamlit as st
from streamlit.navigation.page import StreamlitPage

# pylint: disable=C0301

# ttl of None signifies that it doesn't expire
@st.cache_resource(ttl=None)
def r2r_client():
    """
    Loads the client necessary to interact with the backend.
    R2R has a RESTful API which acts as a server and using this client
    we can interact with it. It sends requests and gets responses behind the scenes.
    """

    # Since when running in compose we can the name of the container as a URL.
    return R2RClient(
        base_url='http://r2r:7272',
        timeout=600
    )

@st.cache_resource
def ollama_client():
    """
    Load Ollama client. 
    Have in mind that this will be containerized and will try to connect to the hosting device.
    `host.docker.internal` will enable exactly that communication from inside the container.
    If you want to run it on the local machine, use `localhost` instead - in the env file.
    """
    return Client(host=st.session_state['ollama_api_base'])

@st.cache_resource
def ollama_options():
    """Load Ollama options. The values here can be tweaked in rag.env file."""
    return Options(
        temperature=st.session_state['temperature'],
        top_p=st.session_state['top_p'],
        top_k=st.session_state['top_k'],
        num_ctx=24000, # This is hard-coded by default.
        format="json", # This should also be json to enforce proper output
    )


def get_pages() -> t.List[StreamlitPage]:
    """Defines main pages of the application."""
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
            page="st_settings.py",
            title="Settings",
            url_path="settings",
            icon=":material/settings:"
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
    pages = get_pages()

    # Register pages. Creates the navigation menu for the application.
    # This page is an entrypoint and as such serves as a page router.
    page = st.navigation(pages)

    # ====== TWEAK VALUES BELOW TO ACHIEVE BEST PERFORMANCE ======

    if "top_k" not in st.session_state:
        st.session_state['top_k'] = int(os.getenv("TOP_K"))

    if "top_p" not in st.session_state:
        st.session_state['top_p'] = float(os.getenv("TOP_P"))

    if "max_tokens" not in st.session_state:
        st.session_state['max_tokens'] = int(os.getenv("MAX_TOKENS"))

    if "chunk_size" not in st.session_state:
        st.session_state['chunk_size'] = int(os.getenv("CHUNK_SIZE"))

    if "temperature" not in st.session_state:
        st.session_state['temperature'] = float(os.getenv("TEMPERATURE"))

    if "chunk_overlap" not in st.session_state:
        st.session_state['chunk_overlap'] = int(os.getenv("CHUNK_OVERLAP"))

    # ====== TWEAK VALUES ABOVE TO ACHIEVE BEST PERFORMANCE ======

    if "exports_dir" not in st.session_state:
        st.session_state['exports_dir'] = "./exports"

    if "conversation_id" not in st.session_state:
        st.session_state['conversation_id'] = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "parent_id" not in st.session_state:
        st.session_state["parent_id"] = None

    if "chat_model" not in st.session_state:
        st.session_state["chat_model"] = os.getenv("CHAT_MODEL")

    if "embedding_model" not in st.session_state:
        st.session_state["embedding_model"] = os.getenv("EMBEDDING_MODEL")

    if "similarity_threshold" not in st.session_state:
        st.session_state["similarity_threshold"] = float(os.getenv("SIMILARITY_THRESHOLD"))

    if "max_relevant_messages" not in st.session_state:
        st.session_state["max_relevant_messages"] = int(os.getenv("MAX_RELEVANT_MESSAGES"))

    if 'ingestion_config' not in st.session_state:
        st.session_state['ingestion_config'] = r2r_client().system.settings().results.config['ingestion']

        # Since the config is a snapshot not an actual instance of the config
        new_ingestion_config = st.session_state['ingestion_config']
        new_ingestion_config['chunk_size'] = st.session_state['chunk_size']
        new_ingestion_config['chunk_overlap'] = st.session_state['chunk_overlap']

        st.session_state['ingestion_config'] = new_ingestion_config

    # Default prompt name that is used by R2R when interacting with /rag endpoint
    if 'selected_prompt' not in st.session_state:
        st.session_state['selected_prompt'] = "rag"

    if 'prompt_template' not in st.session_state:
        st.session_state['prompt_template'] = r2r_client().prompts.retrieve(
            st.session_state['selected_prompt']
        ).results.template

    if 'websearch_api_key' not in st.session_state:
        st.session_state['websearch_api_key'] = None

    if "bearer_token" not in st.session_state:
        st.session_state['bearer_token'] = r2r_client().users.login(
            email = "admin@example.com",
            password = "change_me_immediately"
        ).results.access_token.token

    if "ollama_api_base" not in st.session_state:
        st.session_state['ollama_api_base'] = os.getenv("OLLAMA_API_BASE")

    # Run selected page
    page.run()
