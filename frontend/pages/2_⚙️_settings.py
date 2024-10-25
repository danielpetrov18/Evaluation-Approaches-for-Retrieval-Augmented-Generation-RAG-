# import sys
# import streamlit as st
# from pathlib import Path

# # Add the backend directory to Python path
# backend_dir = Path(__file__).parent.parent / 'backend'
# sys.path.append(str(backend_dir))

# from client import R2RBackend                     

# @st.cache_resource
# def get_r2r_client():
#     config_filepath = backend_dir / 'config' / 'r2r.toml'    
#     return R2RBackend(config_filepath)

# st.sidebar.markdown("# Settings page ❄️")

# # Idea: can use DataFrame to display chunks, metadata etc.
# # Idea: use @st.cache_data when making URL requests for the chunks