"""
This module provides the user with the ability to interact with files and chunks.
Ingestion of files/chunks results into so called Documents. 
Documents are an abstraction that holds the actual data, metadata, might contain summary and so on.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from r2r import R2RException, R2RAsyncClient

class StorageHandler:
    """
    This class enables users to interact with the stored data.
    Additionally deletion, insertion and update are possible.
    """

    def __init__(self, client: R2RAsyncClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        load_dotenv()
        self._file_dir = Path(os.getenv("FILES_DIRECTORY"))
        self._similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD"))

    async def list_documents(self, ids: list[str] = None, offset: int = 0, limit: int = 10):
        """
        Retrieve a list of documents from the R2R service.

        Returns:
            WrappedDocumentsResponse: List of documents in the R2R service.

        Raises:
            R2RException: If there is an error while fetching the list of documents.
            Exception: If an unexpected error occurs.
        """
        try:
            documents = await self._client.documents.list(
                ids=ids,
                offset=offset,
                limit=limit
            )
            return documents
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while listing documents: %s [-]', e)
            raise

    async def ingest_file(self, filepath: str, metadata: dict = None):
        """
        Ingest a file from the local file system into the R2R service. (/backend/data)
        
        Args:
            filepath (str): Path to the file to ingest.
            metadata (Optional[dict], optional): Metadata to associate with the ingested document.

        Returns:
            WrappedIngestionResponse: Ingestion result.

        Raises:
            R2RException: If there is an error while ingesting the document.
            Exception: If an unexpected error occurs.
        """
        try:
            ingestion_mode = 'fast'
            file_extension = str(filepath).rsplit('.', maxsplit=1)[-1]
            if file_extension in ('pdf', 'md'):
                ingestion_mode = 'custom'

            ingestion_result = await self._client.documents.create(
                file_path=filepath,
                ingestion_mode=ingestion_mode,
                metadata=metadata,
                run_with_orchestration=False
            )
            return ingestion_result
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while ingesting document: %s [-]', e)
            raise

    async def ingest_chunks(self, chunks: list[str], metadata: dict = None):
        """
        Ingests a list of text chunks into the R2R service. 

        Args:
            chunks (list[str]): List of text chunks to ingest.
            metadata (dict, optional): Metadata to associate with the ingested document.

        Returns:
            WrappedIngestionResponse: Ingestion result.

        Raises:
            R2RException: If there is an error while ingesting the chunks.
            Exception: If an unexpected error occurs.
        """
        try:
            ingestion_result = await self._client.documents.create(
                chunks=chunks,
                ingestion_mode='fast',
                metadata=metadata,
                run_with_orchestration=False
            )
            return ingestion_result
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while ingesting chunks: %s [-]', e)
            raise

    async def get_document_metadata_by_id(self, doc_id: str):
        """
        Retrieve a document from the R2R service by its id. This doesn't return the actual
        content of the ingested file (assuming it exists). It just gives metadata.

        Args:
            doc_id (str): The id of the document to retrieve.

        Returns:
            WrappedDocumentResponse: The retrieved document.

        Raises:
            R2RException: If there is an error while retrieving the document.
            Exception: If an unexpected error occurs.
        """
        try:
            document = await self._client.documents.retrieve(doc_id)
            return document
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 404) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while retrieving document: %s [-]', e)
            raise

    async def delete_document_by_id(self, doc_id: str):
        """
        Delete a document from the R2R service by its id.

        Args:
            doc_id (str): The id of the document to delete.

        Returns:
            WrappedBooleanResponse: The deletion result.

        Raises:
            R2RException: If there is an error while deleting the document.
            Exception: If an unexpected error occurs.
        """
        try:
            deletion_result = await self._client.documents.delete(doc_id)
            return deletion_result
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while deleting document: %s [-]', e)
            raise

    async def list_document_chunks(self, doc_id: str, offset: int = 0, limit: int = 100):
        """
        Fetches chunks of a document from the R2R service.

        Args:
            doc_id (str): The ID of the document whose chunks are to be fetched.
            offset (int, optional): The starting point for fetching chunks. Defaults to 0.
            limit (int, optional): The maximum number of chunks to fetch. Defaults to 100.

        Returns:
            list: A list of document chunks.

        Raises:
            R2RException: If there is an error while fetching the document chunks.
            Exception: If an unexpected error occurs.
        """

        try:
            chunks = await self._client.documents.list_chunks(
                id=doc_id,
                include_vectors=False,
                offset=offset,
                limit=limit
            )
            return chunks
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 404) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while fetching document chunks: %s [-]', e)
            raise

    async def download_document_content(self, doc_id: str):
        """
        Downloads the original file content of a document.
        For uploaded files, returns the original file with its proper MIME type. 
        For text-only documents, returns the content as plain text.
        
        Args:
            doc_id (str): The id of the document to download.

        Returns:
            Union[BytesIO|str]: The downloaded document content.

        Raises:
            R2RException: If there is an error while downloading the document.
            Exception: If an unexpected error occurs.
        """
        try:
            docs_bytes = await self._client.documents.download(doc_id)
            return docs_bytes
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 404) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while downloading document content: %s [-]', e)
            raise

    async def delete_documents_by_filter(self, filters: dict):
        """
        Delete documents based on provided filters. 

        (Assuming you are using r2r version 3.4.3)
        # class FilterOperator:
        #     EQ = "$eq"
        #     NE = "$ne"
        #     LT = "$lt"
        #     LTE = "$lte"
        #     GT = "$gt"
        #     GTE = "$gte"
        #     IN = "$in"
        #     NIN = "$nin"
        #     LIKE = "$like"
        #     ILIKE = "$ilike"
        #     CONTAINS = "$contains"
        #     AND = "$and"
        #     OR = "$or"
        #     OVERLAP = "$overlap"

        #     SCALAR_OPS = {EQ, NE, LT, LTE, GT, GTE, LIKE, ILIKE}
        #     ARRAY_OPS = {IN, NIN, OVERLAP}
        #     JSON_OPS = {CONTAINS}
        #     LOGICAL_OPS = {AND, OR}

        Args:
            filters (dict): Filter criteria in JSON format.

        Returns:
            WrappedBooleanResponse: Response from the R2R service, containing the status.

        Raises:
            R2RException: If there is an error while deleting the documents.
            Exception: If an unexpected error occurs.
        """
        try:
            deletion_result = await self._client.documents.delete_by_filter(filters)
            return deletion_result
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while deleting documents: %s [-]', e)
            raise

    async def search_document_summaries(self, query: str):
        """
        Searches for document summaries based on a query.

        Args:
            query (str): The query to search for.

        Returns:
            WrappedDocumentSearchResponse: The search response containing all selected data.

        Raises:
            R2RException: If there is an error while searching for summaries.
            Exception: If an unexpected error occurs.
        """
        try:
            summaries_response = await self._client.documents.search(
                query=query,
                search_mode="custom",
                search_settings={
                    "include_metadatas": True,
                    "include_scores": True,
                    "search_strategy": "vanilla",
                    "chunk_settings": {
                        "index_measure": "cosine_distance"
                    }
                }
            )
            return summaries_response
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while searching summaries: %s [-]', e)
            raise

    async def retrieve_similar_chunks(self, query: str, filters: dict = None, limit: int = 10):
        """
        Retrieves similar chunks based on a query and optional filters.

        Args:
            query (str): The query to search for.
            filters (dict, optional): Filter criteria in JSON format. Defaults to None.
            limit (int, optional): The maximum number of chunks to retrieve. Defaults to 10.

        Returns:
            WrappedVectorSearchResponse: The search response containing the similar chunks.

        Raises:
            R2RException: If there is an error while retrieving similar chunks.
            Exception: If an unexpected error occurs.
        """
        try:
            search_settings = {
                "limit": limit,
                "include_metadatas": True,
                "include_scores": True,
                "search_strategy": "vanilla"
            }

            if filters is not None:
                search_settings["filters"] = filters

            similar_chunks = await self._client.chunks.search(
                query=query,
                search_settings=search_settings
            )
            return similar_chunks
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while retrieving similar chunks: %s [-]', e)
            raise

    async def fetch_chunk_by_id(self, chunk_id: str):
        """
        Retrieve a chunk from the R2R service by its id.

        Args:
            chunk_id (str): The id of the chunk to retrieve.

        Returns:
            WrappedChunkResponse: The retrieved chunk.

        Raises:
            R2RException: If there is an error while retrieving the chunk.
            Exception: If an unexpected error occurs.
        """
        try:
            chunk = await self._client.chunks.retrieve(chunk_id)
            return chunk
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 404) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while retrieving chunk: %s [-]', e)
            raise

    async def update_chunk_by_id(self, chunk_id: str, new_text: str, metadata: dict = None):
        """
        Update a chunk in the R2R service by its id.

        Args:
            chunk_id (str): The id of the chunk to update.
            new_text (str): The new text for the chunk.
            metadata (Optional[dict], optional): New metadata for the chunk. Defaults to None.

        Returns:
            WrappedChunkResponse: The updated chunk.

        Raises:
            R2RException: If there is an error while updating the chunk.
            Exception: If an unexpected error occurs.
        """
        try:
            response = await self._client.chunks.update(
                chunk={
                        "id": chunk_id,
                        "text": new_text,
                        "metadata": metadata 
                }
            )
            return response
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while updating chunk: %s [-]', e)
            raise

    async def delete_chunk_by_id(self, chunk_id: str):
        """
        Delete a chunk in the R2R service by its id.

        Args:
            chunk_id (str): The id of the chunk to delete.

        Returns:
            WrappedBooleanResponse: The deletion result.

        Raises:
            R2RException: If there is an error while deleting the chunk.
            Exception: If an unexpected error occurs.
        """
        try:
            response = await self._client.chunks.delete(chunk_id)
            return response
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while deleting a chunk: %s [-]', e)
            raise

    async def list_chunks(self, metadata_filter: dict, filters: dict, include_vect: bool = False):
        """
        Retrieve a list of chunks from the R2R service. 
        The chunks are filtered based on the provided metadata filter and filters.

        Args:
            metadata_filter (dict): Filter criteria for the metadata of the chunks.
            filters (dict): Filter criteria in JSON format.
            include_vectors (bool, optional): If True, the embeddings will be included.

        Returns:
            WrappedChunksResponse: List of chunks in the R2R service.

        Raises:
            R2RException: If there is an error while listing the chunks.
            Exception: If an unexpected error occurs.
        """
        try:
            response = await self._client.chunks.list(
                include_vectors=include_vect,
                metadata_filter=metadata_filter,
                filters=filters
            )
            return response
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while listing chunks: %s [-]', e)
            raise

    async def clean_db(self):
        """
        Cleans the database by deleting all documents from the R2R service.

        This function retrieves the metadata for all documents, extracts their ids,
        and deletes each document by its id.

        NOTE: This is irreversible! Before doing so think about replicating the database.
    
        Raises:
            R2RException: If there is an error while deleting any document.
            Exception: If an unexpected error occurs.
        """

        try:
            docs_metadata = await self.list_documents()
            doc_ids = [doc_metadata["id"] for doc_metadata in docs_metadata]
            for doc_id in doc_ids:
                await self.delete_document_by_id(doc_id)
            self._logger.info("[+] Deleted all files! [+]")
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while cleaning database: %s [-]', e)
            raise
