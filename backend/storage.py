"""
This module provides the user with the ability to interact with files and chunks.
Ingestion of files/chunks results into so called Documents. 
Documents are an abstraction that holds the actual data, metadata, might contain summary and so on.
"""

import io
import os
import logging
from pathlib import Path
from datetime import datetime
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from r2r import R2RException, R2RClient

class Storage:
    """
    This class enables users to interact with the stored data.
    Additionally deletion, insertion and update are possible.
    """

    def __init__(self, client: R2RClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        load_dotenv()
        self._file_dir = Path(os.getenv("FILES_DIRECTORY"))
        self._export_dir = Path(os.getenv("EXPORT_DIRECTORY"))
        self._file_dir.mkdir(parents=True, exist_ok=True)
        self._export_dir.mkdir(parents=True, exist_ok=True)

    def list_documents(self, ids: list[str] = None, offset: int = 0, limit: int = 100):
        """
        Retrieve a list of documents from the R2R service.

        Returns:
            WrappedDocumentsResponse: List of documents in the R2R service.

        Raises:
            R2RException: If there is an error while fetching the list of documents.
            Exception: If an unexpected error occurs.
        """
        try:
            documents = self._client.documents.list(
                ids=ids,
                offset=offset,
                limit=limit
            )
            return documents
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while listing documents: %s [-]', str(e))
            raise

    def ingest_file(self, filepath: str, metadata: dict = None):
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
            ingestion_result = self._client.documents.create(
                file_path=filepath,
                ingestion_mode="fast",
                metadata=metadata,
                run_with_orchestration=True
            )
            return ingestion_result
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while ingesting document: %s [-]', str(e))
            raise

    def ingest_chunks(self, chunks: list[str], metadata: dict = None):
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
            ingestion_result = self._client.documents.create(
                chunks=chunks,
                ingestion_mode='fast',
                metadata=metadata,
                run_with_orchestration=True
            )
            return ingestion_result
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while ingesting chunks: %s [-]', str(e))
            raise

    def export_documents_to_csv(
        self,
        out_path: str|Path,
        bearer_token: str,
        columns: list[str] = None
    ):
        """
        Retrieve a document from the R2R service by its id. This doesn't return the actual
        content of the ingested file (assuming it exists). It just gives metadata.

        Args:
            out_path (str|Path): Filepath to export the documents to.
            bearer_token (str): The bearer token for authorization.
            columns (list[str], optional): List of columns to include in the CSV.
            
        Raises:
            R2RException: If there is during exporting.
            Exception: If an unexpected error occurs.
        """
        try:
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
                "Accept": "text/csv"
            }

            payload = {
                "include_header": "true"
            }

            if columns:
                payload['columns'] = columns
            else:
                payload['columns'] = [
                    "id",
                    "type",
                    "metadata",
                    "title",
                    "ingestion_status",
                    "created_at",
                    "updated_at",
                    "summary",
                    "total_tokens"
                ]

            response = requests.post(
                url="http://localhost:7272/v3/documents/export",
                json=payload,
                headers=headers,
                timeout=5
            )

            if response.status_code != 200:
                raise R2RException(response.text, response.status_code)

            df = pd.read_csv(io.StringIO(response.text))
            if df.shape[0] == 0: # If the dataframe is empty (no rows)
                raise R2RException('No documents found', 404)

            df['metadata'] = df['metadata'].apply(json.loads)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_path = Path(self._export_dir) / f"{out_path}_{timestamp}.csv"

            df.to_csv(out_path, index=False)

        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while exporting documents: %s [-]', str(e))
            raise

    def export_documents_to_zip(
        self,
        *,
        out_path: str|Path,
        bearer_token: str,
        document_ids: list[str] = None,
        start_date: datetime = None,
        end_date: datetime = None
    ):
        """
        Export documents to a zip archive.

        Example:
            from datetime import datetime

            Attributes: year, month, day, hour, minute, second, microsecond,
            start_date = datetime(2025, 3, 7, 10, 0, 0, 0)
            end_date = datetime(2025, 3, 7, 17, 0, 0, 0)

            storage_handler.export_documents_to_zip(
                out_path="testing",
                bearer_token=bearer_token,
                document_ids=None,
                start_date=start_date,
                end_date=end_date
            )
            
        NOTE: Only data that got ingested over the ingest_files endpoint can be exported
              Chunks have correct document_id, however it doesn't work.

        Args:
            out_path (str|Path): Filepath to export the documents to.
            bearer_token (str): The bearer token for authorization.
            document_ids (list[str], optional): List of document IDs to include in the export.
            start_date (datetime, optional): Start date of the export time range.
            end_date (datetime, optional): End date of the export time range.

        Raises:
            R2RException: If there is an error while exporting the documents.
            Exception: If an unexpected error occurs.
        """
        try:
            headers = {
                "Authorization": f"Bearer {bearer_token}"
            }

            params = {}
            if document_ids:
                params['document_ids'] = document_ids
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date

            response = requests.get(
                url="http://localhost:7272/v3/documents/download_zip",
                headers=headers,
                params=params,
                timeout=30
            )

            if response.status_code != 200:
                raise R2RException(response.text, response.status_code)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_path = Path(self._export_dir) / f"{out_path}_{timestamp}.zip"

            with open(out_path, 'wb') as f:
                f.write(response.content)

        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while exporting documents to zip: %s [-]', e)
            raise

    def get_document_metadata_by_id(self, doc_id: str):
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
            document = self._client.documents.retrieve(doc_id)
            return document
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while retrieving document: %s [-]', str(e))
            raise

    def delete_document_by_id(self, doc_id: str):
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
            deletion_result = self._client.documents.delete(doc_id)
            return deletion_result
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while deleting document: %s [-]', str(e))
            raise

    def list_document_chunks(self, doc_id: str, offset: int = 0, limit: int = 100):
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
            chunks = self._client.documents.list_chunks(
                id=doc_id,
                include_vectors=False,
                offset=offset,
                limit=limit
            )
            return chunks
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while fetching doc chunks: %s [-]', str(e))
            raise

    def download_document_content(self, doc_id: str):
        """
        Downloads the original file content of a document.
        For uploaded files, returns the original file with its proper MIME type. 
        For text-only documents, returns the content as plain text.
        Use this if you have ingested a file, you can't get the chunks of a file
        you ingested. You can only get the original file content.
        
        Args:
            doc_id (str): The id of the document to download.

        Returns:
            BytesIO: The downloaded document content.

        Raises:
            R2RException: If there is an error while downloading the document.
            Exception: If an unexpected error occurs.
        """
        try:
            docs_bytes = self._client.documents.download(doc_id)
            return docs_bytes
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while downloading document content: %s [-]', e)
            raise

    def delete_documents_by_filter(self, filters: dict):
        """
        Delete documents based on provided filters. 

        (Assuming you are using r2r version 3.4.4)
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
            deletion_result = self._client.documents.delete_by_filter(filters)
            return deletion_result
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while deleting documents: %s [-]', str(e))
            raise

    def fetch_chunk_by_id(self, chunk_id: str):
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
            chunk = self._client.chunks.retrieve(chunk_id)
            return chunk
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while retrieving chunk: %s [-]', str(e))
            raise

    def update_chunk_by_id(self, chunk_id: str, new_text: str, metadata: dict = None):
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
            response = self._client.chunks.update(
                chunk={
                        "id": chunk_id,
                        "text": new_text,
                        "metadata": metadata 
                }
            )
            return response
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while updating chunk: %s [-]', str(e))
            raise

    def delete_chunk_by_id(self, chunk_id: str):
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
            response = self._client.chunks.delete(chunk_id)
            return response
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while deleting a chunk: %s [-]', str(e))
            raise

    def list_chunks(
        self,
        metadata_filters: dict = None,
        offset: int = 0,
        limit: int = 100,
        filters: dict = None
    ):
        """
        Retrieve a list of chunks from the R2R service.

        Args:
            metadata_filters (dict, optional): Filter to apply based on chunk metadata.
            offset (int, optional): The starting point for fetching chunks.
            limit (int, optional): The maximum number of chunks to fetch.
            filters (dict, optional): Additional filters for chunk retrieval.
            
        Returns:
            list: A list of chunks retrieved from the R2R service.

        Raises:
            ValueError: If the filter is not document_id.
            R2RException: If there is an error while listing the chunks.
            Exception: If an unexpected error occurs.
        """

        try:
            response = self._client.chunks.list(
                include_vectors=False,
                offset=offset,
                limit=limit,
            )

            # Since the frameworks filtering doesn't work properly.
            if filters is not None:
                if filters.keys() != {"document_id"}: # Set matching
                    raise ValueError("Only [document_id] filter is supported!")
                original = response.results
                filtered = [c for c in original if str(c.document_id) == filters["document_id"]]
                if not filtered:
                    self._logger.debug("Invalid filter! Not filtered!")
                response.results = filtered

            # Filter based on metadata
            if metadata_filters is not None:
                original = response.results
                # Go over all the filters and check if they match
                filtered = [
                    c for c in original
                    if all(c.metadata.get(key) == value for key, value in metadata_filters.items())
                ]
                if not filtered:
                    self._logger.debug("Invalid metadata filter! Not filtered!")
                response.results = filtered

            return response
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while listing chunks: %s [-]', str(e))
            raise

    def clean_db(self):
        """
        Cleans the database by deleting all documents from the R2R service.

        This function retrieves the metadata for all documents, extracts their ids,
        and deletes each document by its id.

        NOTE: This is irreversible! Before doing so think about extracting the data.
    
        Raises:
            R2RException: If there is an error while deleting any document.
            Exception: If an unexpected error occurs.
        """

        try:
            docs_metadata = self.list_documents()
            doc_ids = [doc_metadata.id for doc_metadata in docs_metadata.results]
            for doc_id in doc_ids:
                self.delete_document_by_id(doc_id)
            self._logger.debug("[+] Deleted all files! [+]")
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while cleaning database: %s [-]', str(e))
            raise
