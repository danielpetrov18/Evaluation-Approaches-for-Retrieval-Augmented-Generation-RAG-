import streamlit as st

# This will be executed upon starting a new session in a browser tab.
# Once its executed the state will be stored in the session state.
# This servers as a form of persistance between runs.
# Every time the script re-ran it will load the state from the session state.
if "backend" not in st.session_state:
    st.session_state.backend = R2RBackend()

st.sidebar.markdown("# Chat page 🎈")