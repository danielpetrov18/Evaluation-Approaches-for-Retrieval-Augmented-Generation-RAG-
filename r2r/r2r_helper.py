import os
from r2r import R2RClient

class R2RHelper:
    
    def __init__(self):
        self.__client = R2RClient(os.getenv("R2R_HOSTNAME", "http://localhost:7272"))

    def ingest_chunks(self, chunks, metadata=None):
        """
        Ingest chunks of a document into postgres(pgvector).

        Args:
            chunks (list[str]): List of document pieces to ingest.
            metadata (dict[str]): Dictionary of metadata to associate with the document.

        Returns:
            list[dict]: Ingestion results containing message, task_id and a document_id.
        """    
        return self.__client.ingest_chunks(chunks=chunks, metadata=metadata)