"""
This module holds all the available pages of the frontend part of the application.
Every resource defined in the main page can be accessed by all pages.
"""

import os
import typing as t
import streamlit as st
from r2r import R2RClient
from streamlit.navigation.page import StreamlitPage

# pylint: disable=C0301

# ttl of None signifies that it doesn't expire
@st.cache_resource(ttl=None)
def load_client() -> R2RClient:
    """Loads the client necessary to interact with the backend."""
    return R2RClient(
        base_url='http://localhost:7272',
        timeout=600
    )

def get_pages() -> t.List[StreamlitPage]:
    """Defines main pages."""
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

    if "bearer_token" not in st.session_state:
        st.session_state['bearer_token'] = load_client().users.login(
            email = "admin@example.com",
            password = "change_me_immediately"
        ).results.access_token.token

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
        st.session_state['ingestion_config'] = load_client().system.settings().results.config['ingestion']

    # Run selected page
    page.run()
