import os
import tempfile
import streamlit as st
from pathlib import Path
from r2r import R2RException
from app import connect_to_backend

st.markdown("""
    <style>
    .upload-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .file-list {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    .success-message {
        color: #28a745;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .error-message {
        color: #dc3545;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def save_uploaded_files(uploaded_files):
    """
    Save uploaded files to a temporary directory and return their paths.
    """
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    
    return saved_paths

def ingest_files(file_paths):
    """
    Ingest files using the R2R backend client.
    """
    try:
        client = connect_to_backend()
        with st.spinner("Ingesting files..."):
            client.ingest_files(file_paths)
        return True
    except R2RException as r2re:
        st.error(f"Error during ingestion: {str(r2re)}")
        return False
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return False

st.title("📤 File Upload & Ingestion")
    
# Initialize session states
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "displayed_files" not in st.session_state:
    st.session_state.displayed_files = []

# File upload section
with st.container():
    st.markdown("### Upload Documents")
    st.markdown("Select one or multiple files to upload for ingestion into the system.")
    
    # Use a dynamic key for the file uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=["txt", "pdf", "doc", "docx", "md", "json", "html", "htm", "csv", "xlsx", "pptx"],
        help="Supported formats: TXT, PDF, DOC, DOCX, MD, JSON, HTML, CSV, XLSX, PPTX",
        key=f"file_uploader_{st.session_state.upload_key}"
    )
    
    # Update displayed files only when there are uploaded files
    if uploaded_files:
        st.session_state.displayed_files = uploaded_files
        
        # Ingestion controls
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Start Ingestion", type="primary", use_container_width=True):
                file_paths = save_uploaded_files(uploaded_files)
                if ingest_files(file_paths):
                    st.success("🎉 All files have been successfully ingested!")
                    # Clear both states
                    st.session_state.upload_key += 1
                    #st.session_state.displayed_files = []
                    #st.rerun()
        
        with col2:
            if st.button("Clear All", use_container_width=True):
                # Clear both states
                st.session_state.upload_key += 1
                st.session_state.displayed_files = []
                st.rerun()
    elif not uploaded_files and st.session_state.displayed_files:
        # If there are no uploaded files but we have displayed files, clear them
        st.session_state.displayed_files = []

with st.expander("ℹ️ About File Ingestion"):
    st.markdown("""
    - Files are processed and text data is being extracted
    - Files are then split into chunks for efficient retrieval
    - Finally, files are converted into embeddings using pgvector
    - Large files may take longer to process
    - Supported file formats: TXT, JSON, HTML, PDF, DOCX, PPTX, XLSX, CSV, MD
    - For best results, ensure documents are well-formatted
    """)