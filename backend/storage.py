import os
import re
import logging
from uuid import UUID
from pathlib import Path
from urllib.parse import quote
from dotenv import load_dotenv
from typing import Optional, List, Union
from r2r import R2RException, R2RAsyncClient

class StorageHandler:
    
    def __init__(self, client: R2RAsyncClient):  
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        load_dotenv()
        self._file_dir = Path(os.getenv("FILES_DIRECTORY"))
        self._similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD"))
    
    async def list_documents(self, ids: Optional[List[Union[str, UUID]]] = None, offset: int = 0, limit: int = 10) -> List[dict]:      
        """
        Retrieve a list of documents from the R2R service.

        Returns:
            list[dict]: List of documents in the R2R service.

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
            return documents['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while listing documents: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while listing documents: {e} [-]')
            raise Exception(str(e)) from e

    async def ingest_file(self, filepath: str, metadata: Optional[dict] = None):
        """
        Ingest a file from the local file system into the R2R service. The files are located at /backend/data.
        Ingestion mode can be left as the default one (one specified in the toml file).
        Otherwise one needs an OPEN_AI_KEY to be set in the environment variable.
        
        Args:
            filepath (str): Path to the file to ingest.
            metadata (Optional[dict], optional): Metadata to associate with the ingested document. Defaults to None.

        Returns:
            dict: Ingestion result.

        Raises:
            R2RException: If there is an error while ingesting the document.
            Exception: If an unexpected error occurs.
        """
        try:        
            ingestion_result = await self._client.documents.create(
                file_path=filepath,
                metadata=metadata,
                run_with_orchestration=False
            )
            return ingestion_result['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while ingesting document: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while ingesting document: {e} [-]')
            raise Exception(e) from e
 
    async def ingest_chunks(self, chunks: List[str], metadata: Optional[dict] = None) -> dict:    
        """
        Ingests a list of text chunks into the R2R service. The chunks are combined into
        a temporary file, which is then ingested using the specified ingestion mode and metadata.

        Args:
            chunks (List[str]): List of text chunks to ingest.
            ingestion_mode (Optional[str], optional): Ingestion mode to use. Defaults to None.
            metadata (Optional[dict], optional): Metadata to associate with the ingested document. Defaults to None.

        Returns:
            dict: Ingestion result.

        Raises:
            R2RException: If there is an error while ingesting the chunks.
            FileExistsError: If the file already exists.
            Exception: If an unexpected error occurs.
        """
        try:
            filepath = self._save_data_to_disk(
                chunks=chunks, 
                source_url=metadata['source']
            )
            ingestion_result = await self.ingest_file(filepath, metadata)
            return ingestion_result
        except FileExistsError as fe:
            self._logger.error(str(fe))
            return {"error": str(fe)}
        except R2RException as r2re:
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            raise Exception(str(e)) from e
        
    def _save_data_to_disk(self, chunks: list[str], source_url: str) -> str:
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', quote(source_url))
        filename = self._file_dir / f"{safe_filename}.txt"
        
        if filename.exists():
            raise FileExistsError(f"File already exists: {str(filename)}")
        
        combined_content = '\n'.join(chunks)
        with open(file=str(filename), mode='w', encoding='utf-8') as f:
            f.write(combined_content)
        return str(filename)

    async def retrieve_document_by_id(self, doc_id: str) -> dict:   
        try:
            document = await self._client.documents.retrieve(doc_id)
            return document['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while retrieving document: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 404) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while retrieving document: {e} [-]')
            raise Exception(str(e)) from e
        
    async def delete_document_by_id(self, doc_id: str) -> dict:
        try:
            deletion_result = await self._client.documents.delete(doc_id)   
            return deletion_result['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while deleting document: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 404) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while deleting document: {e} [-]')
            raise Exception(str(e)) from e
            
    async def fetch_document_chunks(self, doc_id: str, offset: Optional[int] = 0, limit: Optional[int] = 100, include_vectors: Optional[bool] = False) -> dict:
        try:
            doc_chunks = await self._client.documents.list_chunks(
                id=doc_id,
                include_vectors=include_vectors,
                offset=offset,
                limit=limit
            )
            return doc_chunks['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while fetching document chunks: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 404) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while fetching document chunks: {e} [-]')
            raise Exception(str(e)) from e
        
    async def delete_documents_by_filter(self, filters: dict) -> dict:
        # This can work for example to delete all documents are older than a certain date
        # or with a certain extension
        # https://r2r-docs.sciphi.ai/api-and-sdks/documents/delete-document-by-filter
        # Also check out the structure of each file you get from list_documents()
        try:
            deletion_result = await self._client.documents.delete_by_filter(filters)
            return deletion_result
        except R2RException as r2re:
            err_msg = f'[-] Error while deleting documents: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while deleting documents: {e} [-]')
            raise Exception(str(e)) from e
        
    async def retrieve_similar_chunks(self, query: str, search_settings: Optional[dict] = {"limit": 10}) -> List[dict]:
        try:
            if "include_scores" not in search_settings:
                search_settings["include_scores"] = True
            if "chunks_settings" not in search_settings:
                search_settings["chunks_settings"] = {
                    "index_measure": "cosine_distance",
                    "enabled": True
                }
            if "filters" not in search_settings:
                if "score" not in search_settings["filters"]:
                    search_settings["filters"] = {
                        "score": {
                            "$gte": self._similarity_threshold
                        }
                    }   
            similar_chunks = await self._client.chunks.search(
                query=query,
                search_settings=search_settings
            )
            return similar_chunks['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while retrieving similar chunks: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while retrieving similar chunks: {e} [-]')
            raise Exception(str(e)) from e
        
    async def retrieve_chunk_by_id(self, chunk_id: str) -> dict:
        try:
            chunk = await self._client.chunks.retrieve(chunk_id)
            return chunk['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while retrieving chunk: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 404) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while retrieving chunk: {e} [-]')
            raise Exception(str(e)) from e
        
    async def clean_db(self):    
        """
        Clean the database by deleting all documents in it.
        NOTE: This is irreversible! Before doing so think about replicating the database.
        """
        try:
            docs_metadata = await self.list_documents()
            doc_ids = [doc_metadata["id"] for doc_metadata in docs_metadata]
            for doc_id in doc_ids:
                await self.delete_document_by_id(doc_id)
            self._logger.info(f"[+] Deleted all files! [+]")
        except R2RException as r2re:
            self._logger.warning(f'[-] Error while clearing all files: {r2re} [-]')
            return False
        except Exception as e:
            self._logger.warning(f'[-] Unexpected error while clearing all files: {e} [-]s')