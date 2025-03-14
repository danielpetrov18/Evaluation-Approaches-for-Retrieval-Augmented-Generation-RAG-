"""
This module holds all the available pages of the frontend application.
Every resource defined in the main page can be accessed by all pages
"""

import os
import streamlit as st
from r2r import R2RClient
from streamlit.navigation.page import StreamlitPage

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
            page="st_homepage.py",
            title="Homepage",
            url_path="homepage",
            icon=":material/home:",
            default=True
        ),
        st.Page(
            page="st_conversation.py",
            title="Conversations",
            url_path="conversations",
            icon=":material/forum:"
        ),
        st.Page(
            page="st_storage.py",
            title="Document management",
            url_path="documents",
            icon=":material/docs:"
        ),
        st.Page(
            page="st_settings.py",
            title="Settings",
            url_path="settings",
            icon=":material/settings:"
        ),
    #    st.Page("st_chat.py", title="Chatbot", icon=":material/chat:"),
        st.Page(
            page="st_prompt.py",
            title="Prompts",
            url_path="prompts",
            icon=":material/notes:"
        ),
        st.Page(
            page="st_index.py",
            title="Index Management",
            url_path="index",
            icon=":material/description:"
        )
    ]

if __name__ == "__main__":
    pages = get_pages()

    # Register pages. Creates the navigation menu for the application.
    # This page is an entrypoint and as such serves as a page router.
    page = st.navigation(pages)

    if "token" not in st.session_state:
        bearer_token = load_client().users.login(
            email = "admin@example.com",
            password = "change_me_immediately"
        ).results.access_token.token
        st.session_state["token"] = bearer_token

    if "chunk_size" not in st.session_state:
        st.session_state['chunk_size'] = int(os.getenv("CHUNK_SIZE"))

    if "chunk_overlap" not in st.session_state:
        st.session_state['chunk_overlap'] = int(os.getenv("CHUNK_OVERLAP"))

    if "files_dir" not in st.session_state:
        st.session_state['files_dir'] = os.getenv("FILES_DIRECTORY")

    if "prompts_dir" not in st.session_state:
        st.session_state['prompts_dir'] = os.getenv("PROMPTS_DIRECTORY")

    if "export_dir" not in st.session_state:
        st.session_state['export_dir'] = os.getenv("EXPORT_DIRECTORY")

    if "indices_dir" not in st.session_state:
        st.session_state['indices_dir'] = os.getenv("INDICES_DIRECTORY")

    # Run selected page
    page.run()
