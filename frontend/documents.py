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

def display_documents():
    try:        
        client = connect_to_backend()
        
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

        st.markdown("<div class='document-table'>", unsafe_allow_html=True)
        st.dataframe(
            table_data, 
            #use_container_width=True, 
            #hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    except R2RException as r2re:
        st.error(f"An error occurred while fetching documents: {r2re}")
    except Exception as e:
        st.error(f"An error occurred while fetching documents: {e}")

st.title("📄 Ingested Documents")

# display_button = st.button("Display Documents")

# if display_button:
display_documents()

    #st.subheader("View Document Details")
    
    #document_id = st.text_input("Enter Document ID to view details")
    
    # if document_id:
    #     try:
    #         # Fetch document chunks
    #         chunks = backend_client.document_chunks(document_id)
            
    #         st.write(f"Chunks for Document ID: {document_id}")
    #         st.dataframe(
    #             [{"Chunk ID": chunk.get('chunk_id', 'N/A'), 
    #                 "Text Preview": chunk.get('text', 'N/A')[:100] + '...'} 
    #                 for chunk in chunks],
    #             use_container_width=True
    #         )
    #     except Exception as e:
    #         st.error(f"Could not retrieve document chunks: {e}")