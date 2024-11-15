import os
import logging
from pathlib import Path
from message import Message
from history import ChatHistory
from ollama_helper import OllamaHelper
from r2r import R2RClient, R2RException 
from typing import Iterator, Union, Optional, List

class R2RBackend:

    def __init__(self):
        R2R_HOST = os.getenv("R2R_HOSTNAME", "http://localhost")
        R2R_PORT = os.getenv("R2R_PORT", "7272")
        CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
        EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
        SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
        MAX_HISTORY_SIZE = int(os.getenv("MAX_HISTORY_SIZE", 30))
        MAX_RELEVANT_HISTORY_MESSAGES = int(os.getenv("MAX_RELEVANT_HISTORY_MESSAGES"))
        
        self.__client = R2RClient(f'{R2R_HOST}:{R2R_PORT}')
        self.__logger = logging.getLogger(__name__)
        self.__vector_search_settings = {'index_measure': 'cosine_distance', 'ef_search': '100'}
        self.__history = ChatHistory(max_size=MAX_HISTORY_SIZE)
        self.__ollama_helper = OllamaHelper(CHAT_MODEL, EMBEDDING_MODEL, SIMILARITY_THRESHOLD, MAX_RELEVANT_HISTORY_MESSAGES)

    def health(self) -> str: 
        """
        Check the health of the R2R service.

        Returns:
            str: Returns OK if the service is healthy.
            
        Raises:
            R2RException: If the service is not healthy.
            Exception: If an unexpected error occurs.
        """
        try:
            health_resp = self.__client.health()
            return health_resp['results']['response']
        except R2RException as r2re:
            err_msg = f'[-] Error while checking health: {r2re} [-]'
            self.__logger.error(err_msg)
            raise R2RException(err_msg, 500)
        except Exception as e:
            self.__logger.error(f'[-] Unexpected error while checking health: {e} [-]')
            raise Exception(e)

    def ingest_files(self, filepaths: List[str]): 
            """
            Ingest files into postgres(pgvector). 
            If a document with the same title is already present in the database, nothing gets embedded.
            Invalid filepaths are ignored.

            Args:
                filepaths: Filepaths to ingest.
            
            Raises:
                ValueError: If the path is not a directory.
            """
            if len(filepaths) == 0:
                self.__logger.warning(f'[-] No files found in [{folder_path}]! [-]')
                return
            
            for filepath in filepaths:
                try:
                    self.__client.ingest_files(file_paths=[filepath])
                    self.__logger.info(f'[+] Ingested: {filepath}! [+]')
                except R2RException as r2re:
                    self.__logger.warning(f'[-] [{filepath}] cannot be ingested! {r2re} [-]') 
                except Exception as e:
                    self.__logger.warning(f'[-] Unexpected error when ingesting [{filepath}]: {e} [-]')
        
    def _iterate_over_files(self, folder_path: str | Path) -> Iterator[str]:
        """
        Iterate over all files in a given folder and its sub folders.

        Args:
            folder_path: Path to the folder to iterate over.

        Yields:
            Path to each file.

        Returns:
            Iterator of file paths.
        """
        is_directory = os.path.isdir(folder_path)
        if is_directory is False:
            raise ValueError(f'[-] [{folder_path}] is not a directory! [-]')
        
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        
        return (str(file_path) for file_path in folder_path.rglob('*') if file_path.is_file())
        
    def ingest_chunks(self, chunks: List[dict], metadata: dict[str] = None) -> List[dict]:
        """
        Ingest chunks of a document into postgres(pgvector). 
        If a document with the same title is already present in the database, nothing gets embedded.

        Args:
            chunks (list[str]): List of document pieces to ingest.
            metadata Optional(dict[str]): Dictionary of metadata to associate with the document.

        Raises:
            R2RException: If the ingestion fails.
            Exception: If an unexpected error occurs.

        Returns:
            list[dict]: Ingestion results containing message, task_id and a document_id.
        """    
        try:
            return self.__client.ingest_chunks(chunks=chunks, metadata=metadata)['results']
        except R2RException as r2re:
            err_msg = f"[-] Failed to ingest chunks for: [{metadata['source']}]! {r2re} [-]"
            self.__logger.error(err_msg)
            raise R2RException(err_msg, 500)
        except Exception as e:
            err_msg = f'[-] Unexpected error when ingesting chunks for [{metadata["source"]}]! {e} [-]'
            self.__logger.error(err_msg)
            raise Exception(err_msg)
    
    def update_files(self, filepaths: List[str], document_ids: List[str]):    
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
            raise ValueError("[-] Filepaths and document_ids must have the same length! [-]")
        
        for filepath, document_id in zip(filepaths, document_ids):
            try:
                self.__client.update_files(file_paths=[filepath], document_ids=[document_id])    
                self.__logger.info(f'Updated: {filepath}!')
            except R2RException as r2re:
                self.__logger.warning(f"[-] Failed to update: [{filepath}]! {r2re} [-]")        
            except Exception as e:
                self.__logger.warning(f'[-] Unexpected error when updating file: [{filepath}] - {e} [-]')
    
    # Note: Even if one provides non-existing IDs the function doesn't throw an error. It returns a 200 OK.
    def documents_overview(self, documents_ids: List[str] = None, offset: int = 0, limit: int = 100) -> List[dict]:
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

        Raises:
            R2RException: If the request fails.
            Exception: If an unexpected error occurs.

        Returns:
            list[dict]: List of dictionaries containing chunk_id, text, embeddings, etc.
        """
        try:
            return self.__client.document_chunks(document_id, offset, limit, include_vectors)['results'] 
        except R2RException as r2re:
            err_msg = f"[-] Couldn't get chunks for: [{document_id}]! {r2re} [-]"	
            self.__logger.error(err_msg)
            raise R2RException(err_msg, 500)
        except Exception as e:
            err_msg = f'[-] Unexpected error when getting chunks for [{document_id}]! {e} [-]'
            self.__logger.error(err_msg)
            raise Exception(err_msg)
            
    def delete(self, filters: List[dict]):
        """
        Delete documents from the database based on filters.

        Args:
            filters (list[dict]): List of dictionaries containing filter criteria.
            Example: "document_id": {"$eq": "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"}
        """    
        for filter in filters:
            try:
                self.__client.delete(filter)
                self.__logger.info(f"[+] Deleted a file with following filter: [{filter}]! [+]")
            except R2RException as r2re:
                self.__logger.warning(f"[-] Could not delete a file with following filter: [{filter}]! {r2re} [-]")
            except Exception as e:
                self.__logger.warning(f"[-] Unexpected error when deleting using filter: [{filter}] - {e} [-]")     
       
    def clean_db(self):    
        """
        Clean the database by deleting all documents in it.
        NOTE: This is irreversible! Before doing so think about replicating the database.
        """
        try:
            docs_metadata = self.documents_overview()
            filters = [{"document_id": {"$eq": doc_metadata["id"]}} for doc_metadata in docs_metadata]
            self.delete(filters)
            self.__logger.info(f"[+] Deleted all files! [+]")
        except R2RException as r2re:
            self.__logger.warning(f'[-] Error while clearing all files: {r2re} [-]')
        except Exception as e:
            self.__logger.warning(f'[-] Unexpected error while clearing all files: {e} [-]s')
        
    def _create_message(self, role: str, content: str) -> Message:
        """
        Create a new message with computed embedding.
        
        Args:
            role: Role of the message sender ('user' or 'assistant')
            content: Content of the message
            
        Returns:
            Message object with computed embedding
        """
        embedding = self.__ollama_helper.compute_embedding(content)
        return Message(role=role, content=content, embedding=embedding)
        
    def _get_enhanced_query(self, query: str) -> str:
        """
        Enhance the query with relevant historical context.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query with relevant context
        """
        # Create newest message (from current user query)
        newest_message = self._create_message(role='user', content=query)
        
        # Get all messages from history
        history_messages = self.__history.get_all_messages()
        
        # Also save the latest user query as a new message. We don't want to consider it in the context summary
        self.__history.add_message(newest_message) 
        
        # Find relevant messages based on semantic similarity
        relevant_messages = self.__ollama_helper.get_relevant_messages(
            newest_message.embedding, 
            history=history_messages
        )
        
        if not relevant_messages:
            return query
            
        # Construct prompt for summarizing relevant history
        history_summary_prompt = self.__ollama_helper.construct_history_summary_prompt(
            query=query,
            relevant_history=relevant_messages
        )
        
        # Get summary of relevant context
        context_summary = self.__ollama_helper.summarize_context(history_summary_prompt)
        
        # Enhance query with context
        return self.__ollama_helper.enhance_user_query(query, context_summary)
        
    def prompt_llm(self, query: str, rag_generation_config: dict = None):
        """
        Get relevant answers from the database for a given query, incorporating chat history context.
        Formats the query to work with R2R's default RAG template structure.

        Args:
            query (str): Current user query
            rag_generation_config (dict, optional): Configuration for LLM generation

        Returns:
            str: LLM response
        """           
        try:
            enhanced_query = self._get_enhanced_query(query)        
            response = self.__client.rag(
                query=enhanced_query,
                vector_search_settings=self.__vector_search_settings,
                rag_generation_config=rag_generation_config
            )
            
            data = response['results']['completion']['choices'][0]['message']['content']    
            self.__history.add_message(
                self._create_message(role='assistant', content=data)
            )   
            return data
        except R2RException as r2re:
            self.__logger.error(f'[-] Error when prompting LLM: {r2re} [-]')
            raise R2RException(r2re, 500)
        except Exception as e:
            err_msg = f'[-] Unexpected error when prompting LLM: {e} [-]'
            self.__logger.error(err_msg)
            raise Exception(err_msg)
        
    def clear_history(self):
        self.__logger.info('[-] Clearing chat history! [-]')
        self.__history.clear()