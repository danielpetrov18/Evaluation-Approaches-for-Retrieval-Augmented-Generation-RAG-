"""
This module holds all the available pages of the frontend application.
"""

import asyncio
from r2r import R2RAsyncClient
import streamlit as st
from streamlit.navigation.page import StreamlitPage

# Every resource defined in the main page can be accessed by all pages
def load_client() -> R2RAsyncClient:
    """Loads the client necessary to interact with the backend."""
    return R2RAsyncClient('http://localhost:7272')

def run_coroutine(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)

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
    #    st.Page("st_storage.py", title="Document management", url_path="documents", icon=":material/docs:"),
        st.Page(
            page="st_settings.py",
            title="Settings",
            url_path="settings",
            icon=":material/settings:"
        ),
    #    st.Page("st_chat.py", title="Chatbot", icon=":material/chat:"),  
    #    st.Page("st_prompt.py", title="Prompts", url_path="prompts", icon=":material/notes:"),
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

    # Run selected page
    page.run()
