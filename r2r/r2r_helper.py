import os
import json
from r2r import R2RClient

class R2RHelper:
    
    def __init__(self):
        self.__client = R2RClient(os.getenv("R2R_HOSTNAME", "http://localhost:7272"))

    def ingest_chunks(self, chunks, document_id=None, metadata=None):
        """
        Ingest chunks of a document into postgres(pgvector).

        Args:
            chunks (list[str]): List of document pieces to ingest.
            metadata (dict[str]): Dictionary of metadata to associate with the document.

        Returns:
            list[dict]: Ingestion results containing message, task_id and a document_id.
        """    
        metadata_str = json.dumps(metadata) # If not converted this way I get 422 unprocessable entity.
        return self.__client.ingest_chunks(chunks=chunks, document_id=document_id, metadata=metadata_str)
    
    def documents_overview(self, documents_ids=None):
        """
        Get an overview of documents in the database.

        Args:
            documents_ids (list[str], optional): List of document IDs to get overview of. If not provided, overview of all documents is returned.

        Returns:
            list[dict]: List of dictionaries containing name, id, vector_size, num_docs, and date_created for each document.
        """
        return self.__client.documents_overview(documents_ids)["results"]