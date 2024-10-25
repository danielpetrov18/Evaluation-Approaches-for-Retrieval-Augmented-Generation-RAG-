import os
import logging
from r2r import R2RException, R2RClient

class R2RBackend:

    def __init__(self, config_path: str = './config.toml'):
        R2R_HOST = os.getenv("R2R_HOSTNAME", "http://localhost")
        R2R_PORT = os.getenv("R2R_PORT", "7272")
        self.__client = R2RClient(f'{R2R_HOST}:{R2R_PORT}')
        self.__logger = logging.getLogger(__name__)
        self.__vector_search_settings = { 
                                         'index_measure': 'cosine_distance',
                                         'ef_search': '100' # Increases accuracy and decreases speed
        }

    def health(self) -> dict: 
        """
        Check the health of the R2R service.

        Returns:
            dict[str]: Dictionary containing information about the health of the service. Should contain OK.
        """
        try:
            health_resp = self.__client.health()
            return health_resp['results']['response']
        except R2RException as r2re:
            self.__logger.error(r2re)
            raise R2RException(r2re)
        except Exception as e:
            self.__logger.error(e)
            raise Exception(e)

    def ingest_files(self, filepaths: list[str]) -> None: 
            """
            Ingest files into postgres(pgvector). 
            If a document with the same title is already present in the database, nothing gets embedded.
            Invalid filepaths are ignored.

            Args:
                file_paths (list[str]): List of file paths to ingest. Should be locally available.
                
            Returns:
                None
            """
            for filepath in filepaths:
                try:
                    self.__client.ingest_files(file_paths=[filepath])
                    self.__logger.info(f'Ingested: {filepath}!')
                except R2RException as r2re:
                    err_msg = f'[{filepath}] has already been ingested!'
                    # Just a warning because I don't want to stop the ingestion of all files.
                    # If some fail one can retry later.
                    self.__logger.warn(err_msg) 
                except Exception as e:
                    self.__logger.warn(e)
        
    def ingest_chunks(self, chunks: list[dict], metadata: dict[str] = None) -> list[dict]:
        """
        Ingest chunks of a document into postgres(pgvector). 
        If a document with the same title is already present in the database, nothing gets embedded.

        Args:
            chunks (list[str]): List of document pieces to ingest.
            metadata Optional(dict[str]): Dictionary of metadata to associate with the document.

        Returns:
            list[dict]: Ingestion results containing message, task_id and a document_id.
        """    
        try:
            return self.__client.ingest_chunks(chunks=chunks, metadata=metadata)['results']
        except R2RException as r2re:
            err_msg = f"Failed to ingest chunks for: [{metadata['source']}]!"
            self.__logger.error(err_msg)
            raise R2RException(err_msg)
        except Exception as e:
            self.__logger.error(e)
            raise Exception(e)
    
    def update_files(self, filepaths: list[str], document_ids: list[str]) -> int:    
        """
        Update files in the database. If a document with the same title is already present in the database, it gets updated.

        Args:
            file_paths (list[str]): List of file paths to update.
            document_ids (list[str]): List of document IDs to associate with the updated files.

        Raises:
            ValueError: If the lengths of filepaths and document_ids are not equal.    

        Returns:
            int: Files updated.
        """
        if len(filepaths) != len(document_ids):
            raise ValueError("Filepaths and document_ids must have the same length!")
        
        files_updated = 0
        for filepath, document_id in zip(filepaths, document_ids):
            try:
                self.__client.update_files(filepaths, document_ids)    
                files_updated += 1
            except R2RException as r2re:
                err_msg = f"Failed to update: [{filepath}]!"
                self.__logger.warn(err_msg)
            except Exception as e:
                self.__logger.warn(e)
            
        return files_updated
    
    def documents_overview(self, documents_ids: list[str] = None, offset: int = 0, limit: int = 100) -> list[dict]:
        """
        Get an overview of documents in the database.

        Args:
            documents_ids (list[str], optional): List of document IDs to get overview of. If not provided, overview of all documents is returned.

        Returns:
            list[dict]: List of dictionaries containing document_id, version, title, etc.
        """
        return self.__client.documents_overview(documents_ids, offset, limit)["results"]
        
    def document_chunks(self, document_id: str, offset: int = 0, limit: int = 100, include_vectors: bool = False):  
        """
        Get chunks of a document.

        Args:
            document_id (str): ID of the document to get chunks of.
            offset Optional(int): Offset of the first chunk to return. Defaults to 0.
            limit Optional(int): Maximum number of chunks to return. Defaults to 100.
            include_vectors Optional(bool): Whether to include embeddings in the returned chunks. Defaults to False.

        Returns:
            list[dict]: List of dictionaries containing chunk_id, text, embeddings, etc.
        """
        try:
            return self.__client.document_chunks(document_id)['results']
        except R2RException as r2re:
            err_msg = f'Coulndt get chunks for: [{document_id}]!'	
            self.__logger.error(err_msg)
            raise R2RException(err_msg)
        except Exception as e:
            self.__logger.error(e)
            raise Exception(e)
            
    def delete(self, filters: list[dict]) -> None:
        """
        Delete documents from the database based on filters.

        Args:
            filters (list[dict]): List of dictionaries containing filter criteria.
            Example: "document_id": {"$eq": "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"}
        Returns:
            None
        """    
        files_deleted = 0
        for filter in filters:
            try:
                self.__client.delete(filter)
                files_deleted += 1
            except R2RException as r2re:
                err_msg = f"Could not delete a file with following filter: {filter}!"
                self.__logger.warn(err_msg)
            except Exception as e:
                self.__logger.warn(e)
            
        return files_deleted
    
    def clean_db(self) -> None:    
        """
        Clean the database by deleting all documents in it.
        NOTE: This is irreversible! Before doing so think about replicating the database.

        Returns:
            None
        """
        try:
            docs_metadata = self.documents_overview()
            filters = [{"document_id": {"$eq": doc_metadata["document_id"]}} for doc_metadata in docs_metadata]
            self.delete(filters)
        except R2RException as r2re:
            self.__logger.warn(r2re)
        except Exception as e:
            self.__logger.warn(e)
        
    def rag(self, query: str, rag_generation_config: dict = None) -> str:
        """
        Get relevant answers from the database for a given query, considering chat history if present.

        Args:
            query (str): Current query to get relevant answers for.
            rag_generation_config (dict, optional): Configuration for the LLM generation including:
                - temperature: float between 0 and 1
                - top_p: float between 0 and 1
                - max_tokens: integer for maximum response length

        Returns:
            dict: Response containing answer text, status, etc.
        """
        try:
            resp = self.__client.rag(
                query=query,
                vector_search_settings=self.__vector_search_settings,
                rag_generation_config=rag_generation_config
            )   
            return resp['results']['completion']['choices'][0]['message']['content']
        except R2RException as r2re:
            self.__logger.error(r2re)
            raise R2RException(r2re)
        except Exception as e:
            self.__logger.error(e)
            raise Exception(e)