"""
This module holds all the available pages of the frontend application.
Every resource defined in the main page can be accessed by all pages.
"""

import os
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

def get_pages() -> list[StreamlitPage]:
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

    if "chunk_size" not in st.session_state:
        st.session_state['chunk_size'] = int(os.getenv("CHUNK_SIZE"))

    if "chunk_overlap" not in st.session_state:
        st.session_state['chunk_overlap'] = int(os.getenv("CHUNK_OVERLAP"))

    if "exports_dir" not in st.session_state:
        st.session_state['exports_dir'] = os.getenv("EXPORTS_DIRECTORY")

    if "bearer_token" not in st.session_state:
        st.session_state['bearer_token'] = load_client().users.login(
            email = "admin@example.com",
            password = "change_me_immediately"
        ).results.access_token.token

    if "conversation_id" not in st.session_state:
        st.session_state['conversation_id'] = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_model" not in st.session_state:
        st.session_state["chat_model"] = os.getenv("CHAT_MODEL")

    if "embedding_model" not in st.session_state:
        st.session_state["embedding_model"] = os.getenv("EMBEDDING_MODEL")

    if "similarity_threshold" not in st.session_state:
        st.session_state["similarity_threshold"] = float(os.getenv("SIMILARITY_THRESHOLD"))

    if "max_relevant_messages" not in st.session_state:
        st.session_state["max_relevant_messages"] = int(os.getenv("MAX_RELEVANT_MESSAGES"))

    if "rag_generation_config" not in st.session_state:
        st.session_state["rag_generation_config"] = load_client().system.settings().results.config['completion']['generation_config']

    # Run selected page
    page.run()
