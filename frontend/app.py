import streamlit as st
from r2r import R2RAsyncClient

def load_client() -> R2RAsyncClient:
    return R2RAsyncClient('http://localhost:7272')
    
# Create pages
pages = [
    st.Page("st_welcome.py", title="Welcome page", url_path="welcome", icon=":material/home:"),
    st.Page("st_storage.py", title="Document management", url_path="documents", icon=":material/docs:"),
    st.Page("st_settings.py", title="Settings", url_path="settings", icon=":material/settings:"),
    st.Page("st_chat.py", title="Chatbot", icon=":material/chat:"),  
    st.Page("st_prompt.py", title="Prompts", url_path="prompts", icon=":material/notes:"),
    st.Page("st_index.py", title="Index Management", url_path="index", icon=":material/description:"),
] 

# Register pages. Creates the navigation menu for the application. 
# This page is an entrypoint and as such serves as a page router.
page = st.navigation(pages) 

# Run selected page
page.run()