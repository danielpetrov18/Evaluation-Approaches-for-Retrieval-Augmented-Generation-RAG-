import sys
import streamlit as st
from pathlib import Path

backend_dir = Path(__file__).parent.parent / 'backend' # Add the path of the backend client.
sys.path.append(str(backend_dir)) # When a package is being imported python first looks at the sys.builtins path first. Then this path.
from client import R2RBackend
from stream_handler import R2RStreamHandler

@st.cache_resource
def load_client():
    return R2RBackend()

@st.cache_resource
def stream_handler():
    return R2RStreamHandler()

# Create pages
pages = [
    st.Page("chat.py", title="Chatbot", default=True, icon=":material/chat:"),  
    st.Page("uploads.py", title="Ingest files", url_path="uploads", icon=":material/upload:"),
    st.Page("documents.py", title="Documents Overview", url_path="documents", icon=":material/description:")
] 

# Register pages. Creates the navigation menu for the application. 
# This page is an entrypoint and as such serves as a page router.
page = st.navigation(pages) 

# Run selected page
page.run()