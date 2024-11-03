import time
import streamlit as st
from r2r import R2RException
from app import connect_to_backend

st.markdown("""
    <style>
    .document-table {
        width: 100%;
        border-collapse: collapse;
    }
    .document-table th {
        background-color: #f0f2f6;
        color: #333;
        font-weight: bold;
        padding: 10px;
        border: 1px solid #ddd;
    }
    .document-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .document-table tr:hover {
        background-color: #f5f5f5;
    }
    .document-table td {
        padding: 10px;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

def display_documents(client):
    try:        
        documents = client.documents_overview()
        if not documents:
            st.info("No documents found.")
            return

        # Prepare data for display
        table_data = []
        for doc in documents:
            metadata = doc['metadata']
            table_data.append({
                "Document ID": doc['id'],
                "Title": doc.get('title', 'N/A'),
                "Type": doc['type'],
                "Created At": doc['created_at'],
                "Ingestion Status": doc['ingestion_status'],
                "Version": metadata['version'],
                "Source": metadata.get('source', 'N/A')
            })

        # Display documents with selection
        event = st.dataframe(
            table_data, 
            key="documents_table",
            on_select="rerun", # If one selects a row the script will rerun
            selection_mode="single-row"
        )

        # Check if a document is selected
        if event.selection.rows:
            # Get the selected document ID
            selected_row = event.selection.rows[0]
            selected_doc_id = table_data[selected_row]["Document ID"]
            
            # Columns for actions
            col1, col2 = st.columns(2)
            
            with col1:
                # View Chunks Button
                if st.button("View Document Chunks"):
                    try:
                        # Fetch document chunks
                        chunks = client.document_chunks(selected_doc_id)
                        
                        st.subheader(f"Chunks for Document ID: {selected_doc_id}")
                        chunk_data = [
                            {
                                "Chunk ID": chunk.get('chunk_id', 'N/A'), 
                                "Text Preview": chunk.get('text', 'N/A')[:200] + '...'
                            } 
                            for chunk in chunks
                        ]
                        
                        st.dataframe(chunk_data, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not retrieve document chunks: {e}")
            
            with col2:
                # Delete Document Button
                delete_confirm = st.checkbox(f"Confirm deletion of document {selected_doc_id}")
                if delete_confirm and st.button("Delete Document"):
                    try:
                        # Prepare filter for deletion
                        delete_filter = [{"document_id": {"$eq": selected_doc_id}}]
                        
                        # Attempt to delete the document
                        deleted_count = client.delete(delete_filter)
                        
                        if deleted_count > 0:
                            st.success(f"Document {selected_doc_id} deleted successfully!", icon="✅")
                            st.rerun()  # Refresh the page
                        else:
                            st.warning("No document was deleted.", icon="⚠️")
                    except Exception as e:
                        st.error(f"Could not delete document: {e}")

    except R2RException as r2re:
        st.error(f"An error occurred while fetching documents: {r2re}")
    except Exception as e:
        st.error(f"An error occurred while fetching documents: {e}")

st.title("📄 Ingested Documents")

client = connect_to_backend()

with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    delete_btn = st.button("Delete all documents", type="primary", use_container_width=True, help="Delete all ingested data.")
    if delete_btn:
        try:
            client.clean_db()
            st.success("All documents deleted successfully!", icon="✅")
            time.sleep(2)
            st.rerun()
        except R2RException as r2re:
            st.warning(f"An error occurred while deleting documents: {r2re}", icon="⚠️")
        except Exception as e:
            st.warning(f"An error occurred while deleting documents: {e}", icon="⚠️")

# Always display documents
display_documents(client)