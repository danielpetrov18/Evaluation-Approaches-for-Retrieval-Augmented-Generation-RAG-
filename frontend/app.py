import sys
import streamlit as st
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))
from client import R2RBackend

# All widgets and resources created here will be shared across all pages.
# https://docs.streamlit.io/develop/concepts/multipage-apps/overview
@st.cache_resource(show_spinner='Connecting to backend ...')
def connect_to_backend():
    client = R2RBackend()
    resp = client.health()
    if resp == 'ok':
        return client
    else:
        st.error(f"An error occurred while connecting to the backend: {resp}")

# Create pages
pages = [
    st.Page("chat.py", title="Chat", url_path="chat", icon=":material/chat:"),  
    st.Page("documents.py", title="Documents", url_path="documents", icon=":material/description:"),
    st.Page("settings.py", title="Settings", url_path="settings", icon=":material/settings:")
] 

# st.page_link()

# Register pages. Creates the navigation menu for the application. 
# This page is an entrypoint and as such serves as a page router.
page = st.navigation(pages) 

# Run selected page
page.run()