import streamlit as st

# Create pages
pages = [
    st.Page("chat.py", title="Chat", url_path="chat"),  
    st.Page("settings.py", title="Settings", url_path="settings")
] 

# Register pages
page = st.navigation(pages) 

# Run selected page
page.run()