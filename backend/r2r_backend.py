import os
import json
import logging
from pathlib import Path
from r2r import R2RException, R2RClient, R2RConfig
from helper.io_helper import load_prompt, parse_r2r_error

class R2RBackend:

    def __init__(self, config_path: str = './config/r2r.toml'):
        R2R_HOST = os.getenv("R2R_HOSTNAME", "http://localhost")
        R2R_PORT = os.getenv("R2R_PORT", "7272")
        self.__client = R2RClient(f'{R2R_HOST}:{R2R_PORT}')
        self.__logger = logging.getLogger(__name__)
        self.__vector_search_settings = { 'index_measure': 'cosine_distance' }
        self.__prompt_name = 'custom_rag' # Check out the chat_templates folder.
        self.__config_path = config_path
        self.__set_custom_prompt_template()

    def health(self) -> dict: 
        """
        Check the health of the R2R service.

        Returns:
            dict[str]: Dictionary containing information about the health of the service.
        """
        try:
            return self.__client.health()["results"]
        except R2RException as r2re:
            err_msg = parse_r2r_error(r2re)
            self.__logger.error(err_msg)
            raise Exception(err_msg)
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
                    self.__logger.debug(f'Ingested: {filepath} ...')
                except R2RException as r2re:
                    err_msg = parse_r2r_error(r2re)
                    self.__logger.error(err_msg)
                    raise Exception(err_msg)
                except Exception as e:
                    self.__logger.error(e)
                    raise Exception(e)
        
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
            err_msg = parse_r2r_error(r2re)
            self.__logger.error(err_msg)
            raise Exception(err_msg)
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
                err_msg = parse_r2r_error(r2re)
                self.__logger.error(err_msg)
                raise Exception(err_msg)
            except Exception as e:
                self.__logger.error(e)
                raise Exception(e)
            
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
            err_msg = parse_r2r_error(r2re)
            self.__logger.error(err_msg)
            raise Exception(err_msg)
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
                err_msg = parse_r2r_error(r2re)
                self.__logger.error(err_msg)
                raise Exception(err_msg)
            except Exception as e:
                self.__logger.error(e)
                raise Exception(e)
            
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
            err_msg = parse_r2r_error(r2re)
            self.__logger.error(err_msg)
            raise Exception(err_msg)
        except Exception as e:
            self.__logger.error(e)
            raise Exception(e)        
        
    def rag(self, query: str) -> dict: 
        """
        Get relevant answers from the database for a given query.

        Args:
            query (str): Query to get relevant answers for.

        Returns:
            list[dict]: List of dictionaries containing answer text, embeddings, etc.
        """
        try:
            resp = self.__client.rag(
                            query=query, 
                            vector_search_settings=self.__vector_search_settings
            )
            return resp['results']
        except R2RException as r2re:
            err_msg = parse_r2r_error(r2re)
            self.__logger.error(err_msg)
            raise Exception(err_msg)
        except Exception as e:
            self.__logger.error(e)
            raise Exception(e)
    
    def add_prompt(self, prompt_name: str, template: str, input_types: dict) -> dict:
        try:
            return self.__client.add_prompt(name=prompt_name, template=template, input_types=input_types)['results']
        except R2RException as r2re:
            err_msg = parse_r2r_error(r2re)
            self.__logger.error(err_msg)
            raise Exception(err_msg)
        except Exception as e:
            self.__logger.error(e)
            raise Exception(e)
        
    def get_all_prompts(self) -> list[dict]:
        try:
            return self.__client.get_all_prompts()['results']
        except R2RException as r2re:
            err_msg = parse_r2r_error(r2re)
            self.__logger.error(err_msg)
            raise Exception(err_msg)
        except Exception as e:
            self.__logger.error(e)
            raise Exception(e)
    
    def update_prompt(self, name: str, template: str, input_types: dict) -> dict:
        try:
            return self.__client.update_prompt(name=prompt_name, template=template, input_types=input_types)    
        except R2RException as r2re:
            err_msg = parse_r2r_error(r2re)
            self.__logger.error(err_msg)
            raise Exception(err_msg)
        except Exception as e:
            self.__logger.error(e)
            raise Exception(e)
    
    def delete_prompt(self, prompt_name: str):
        try:
            return self.__client.delete_prompt(prompt_name)
        except R2RException as r2re:
            err_msg = parse_r2r_error(r2re)
            self.__logger.error(err_msg)
            raise Exception(err_msg)
        except Exception as e:
            self.__logger.error(e)
            raise Exception(e)
        
    def __set_custom_prompt_template(self):
        is_prompt_present = self.__check_prompt_exists()
        if not is_prompt_present:
            prompt_data = load_prompt(Path('chat_templates'), self.__prompt_name)
            template = prompt_data['template']
            input_types = prompt_data['input_types']
            self.add_prompt(prompt_name=self.__prompt_name, template=template, input_types=input_types)
        
        self.__app_config = R2RConfig.from_toml(self.__config_path)
        self.__app_config.prompt.default_task_name = self.__prompt_name # This sets the prompt to my one
            
    def __check_prompt_exists(self):
        prompts = self.get_all_prompts()['prompts']
        for key, _ in prompts.items():
            if key == self.__prompt_name:
                return True
        return False