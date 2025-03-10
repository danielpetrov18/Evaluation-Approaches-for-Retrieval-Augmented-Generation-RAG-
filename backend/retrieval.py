"""
This module holds the logic for interacting with the underlying LLM.
"""

import logging
from datetime import datetime
from r2r import R2RAsyncClient, R2RException
from shared.abstractions.search import SearchMode

class RetrievalHandler:
    """
    One can use this class to perform semantic similarity retrieval, rag, agent based interactions.
    """

    def __init__(self, client: R2RAsyncClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

    async def search(
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
                "include_metadatas": True,
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

            search_resp = await self._client.retrieval.search(
                query=query,
                search_mode=SearchMode.custom,
                search_settings=search_settings
            )
            return search_resp
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while performing semantic search: %s [-]', e)
            raise

    async def rag(
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
                "include_metadatas": True,
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

            rag_resp = await self._client.retrieval.rag(
                query=query,
                search_mode=SearchMode.custom,
                search_settings=search_settings,
                include_title_if_available=True
            )
            return rag_resp
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error during RAG query: %s [-]', e)
            raise

    async def rag_agent(
        self,
        query: str,
        conversation_id: str,
        filters: dict = None,
        limit: int = 10,
        offset: int = 0,
        ef_search: int = 100,
        probes: int = 10
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
            ef_search (int, optional): Size of the dynamic candidate list for HNSW index search. 
                Higher increases accuracy but decreases speed.
            probes (int, optional): Number of ivfflat index lists to query. 
                Higher increases accuracy but decreases speed.

        Returns:
            WrappedAgentResponse: The search response containing the results.

        Raises:
            R2RException: If there is an error while performing the search.
            Exception: If an unexpected error occurs.
        """
        try:
            message = {
                "role": "user",
                "content": query,
                "metadata": {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }

            search_settings = {
                "use_semantic_search": True,
                "limit": limit,
                "offset": offset,
                #"include_metadatas": True,
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

            agent_resp = await self._client.retrieval.agent(
                message=message,
                search_mode=SearchMode.custom,
                search_settings=search_settings,
                include_title_if_available=True,
                conversation_id=conversation_id
            )
            return agent_resp
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error during RAG agent query: %s [-]', e)
            raise
