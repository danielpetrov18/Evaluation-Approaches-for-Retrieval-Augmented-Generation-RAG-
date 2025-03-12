"""
This module holds the logic for interacting with the underlying LLM.
"""

import logging
from datetime import datetime
from r2r import R2RClient, R2RException

class Retrieval:
    """
    One can use this class to perform semantic similarity retrieval, rag, agent based interactions.
    """

    def __init__(self, client: R2RClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

    def search(
        self,
        query: str,
        filters: dict = None,
        limit: int = 10,
        offset: int = 0,
        ef_search: int = 100,
        probes: int = 10
    ):
        """
        Perform a semantic search using the LLM.

        Args:
            query (str): The query to search for.
            filters (dict, optional): Filters to apply to the search results.
            limit (int, optional): The maximum number of results to return. Defaults to 10.
            offset (int, optional): The offset for pagination. Defaults to 0.
            ef_search (int, optional): Size of the dynamic candidate list for HNSW index search. 
                Higher increases accuracy but decreases speed.
            probes (int, optional): Number of ivfflat index lists to query. 
                Higher increases accuracy but decreases speed.

        Returns:
            WrappedVectorSearchResponse: The search response containing the results.

        Raises:
            R2RException: If there is an error while performing the search.
            Exception: If an unexpected error occurs.
        """
        try:
            search_settings = {
                "use_semantic_search": True,
                "limit": limit,
                "offset": offset,
                "include_metadatas": False,
                "include_scores": True,
                "search_strategy": "vanilla",
                "chunk_settings": {
                    "index_measure": "cosine_distance",
                    "probes": probes,
                    "ef_search": ef_search,
                    "enabled": True
                }
            }

            if filters:
                search_settings["filters"] = filters

            search_resp = self._client.retrieval.search(
                query=query,
                search_mode="custom",
                search_settings=search_settings
            )
            return search_resp
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while performing semantic search: %s [-]', e)
            raise

    def rag(
        self,
        query: str,
        filters: dict = None,
        limit: int = 10,
        offset: int = 0,
        ef_search: int = 100,
        probes: int = 10
    ):
        """
        Perform a RAG-based query.

        Args:
            query (str): The query for RAG.
            filters (dict, optional): Filters to apply to the search results.
            limit (int, optional): The maximum number of results to return. Defaults to 10.
            offset (int, optional): The offset for pagination. Defaults to 0.
            ef_search (int, optional): Size of the dynamic candidate list for HNSW index search. 
                Higher increases accuracy but decreases speed.
            probes (int, optional): Number of ivfflat index lists to query. 
                Higher increases accuracy but decreases speed.

        Returns:
            WrappedRAGResponse: The search response containing the results.

        Raises:
            R2RException: If there is an error while performing the search.
            Exception: If an unexpected error occurs.
        """
        try:
            search_settings = {
                "use_semantic_search": True,
                "limit": limit,
                "offset": offset,
                "include_scores": True,
                "include_metadatas": False,
                "search_strategy": "vanilla",
                "chunk_settings": {
                    "index_measure": "cosine_distance",
                    "probes": probes,
                    "ef_search": ef_search,
                    "enabled": True
                }
            }

            if filters:
                search_settings["filters"] = filters

            rag_resp = self._client.retrieval.rag(
                query=query,
                search_mode="custom",
                search_settings=search_settings,
                include_title_if_available=True
            )
            return rag_resp
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error during RAG query: %s [-]', e)
            raise

    def rag_agent(
        self,
        query: str,
        conversation_id: str = None,
        filters: dict = None,
        limit: int = 10,
        offset: int = 0
    ):
        """
        Perform a RAG agent query. Allows the user to interact with an agent who keeps track
            of previous interactions and context. One can ask follow-up questions.

        Args:
            query (str): The query to search for.
            conversation_id (str): The ID of the conversation to query.
            filters (dict, optional): Filters to apply to the search results.
            limit (int, optional): The maximum number of results to return. Defaults to 10.
            offset (int, optional): The offset for pagination. Defaults to 0.

        Returns:
            WrappedAgentResponse: The search response containing the results.

        Raises:
            R2RException: If there is an error while performing the search.
            Exception: If an unexpected error occurs.
        """
        try:
            response = self._rag_request(
                query = query,
                conversation_id = conversation_id,
                limit = limit,
                offset = offset,
                filters = filters
            )
            return response
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error during RAG agent query: %s [-]', e)
            raise

    def arag_agent(
        self,
        query: str,
        conversation_id: str = None,
        filters: dict = None,
        limit: int = 10,
        offset: int = 0
    ):
        """
        Perform an Agent-based query with streaming enabled.

        Args:
            query (str): The query to search for.
            conversation_id (str): The ID of the conversation to query.
            filters (dict, optional): Filters to apply to the search results.
            limit (int, optional): The maximum number of results to return. Defaults to 10.
            offset (int, optional): The offset for pagination. Defaults to 0.

        Returns:
            WrappedAgentResponse: The search response containing the results.

        Raises:
            R2RException: If there is an error while performing the search.
            Exception: If an unexpected error occurs.
        """
        try:
            response_gen = self._rag_request(
                query = query,
                stream = True,
                conversation_id = conversation_id,
                limit = limit,
                offset = offset,
                filters = filters
            )
            return response_gen
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error during RAG agent stream query: %s [-]', e)
            raise

    def _rag_request(
        self,
        query: str,
        stream: bool = False,
        conversation_id: str = None,
        limit: int = 10,
        offset: int = 0,
        filters: dict = None
    ):
        msg = {
            "role": "user",
            "content": query,
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        gen_conf = self._client.system.settings().results.config['completion']['generation_config']
        if stream:
            gen_conf['stream'] = True

        search_settings = {
            "use_semantic_search": True,
            "limit": limit,
            "offset": offset,
            "include_metadatas": False,
            "include_scores": True,
            "search_strategy": "vanilla",
        }

        if filters:
            search_settings["filters"] = filters

        return self._client.retrieval.agent(
            message = msg,
            rag_generation_config = gen_conf,
            search_mode = "custom",
            search_settings = search_settings,
            include_title_if_available = True,
            conversation_id = conversation_id
        )

    def embed(self, query: str):
        """
        Generate an embedding for a given query using the retrieval client.

        Args:
            query (str): The query for which to generate the embedding.

        Returns:
            EmbeddingResponse: The response containing the embedding result.

        Raises:
            R2RException: If there is an error while generating the embedding.
            Exception: If an unexpected error occurs.
        """

        try:
            embedding_resp = self._client.retrieval.embedding(query)
            return embedding_resp
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error during RAG agent query: %s [-]', e)
            raise
