import os
import logging
from message import Message
from dotenv import load_dotenv
from history import ChatHistory
from ollama_helper import OllamaHelper
from r2r import R2RException, R2RAsyncClient
from typing import Optional, List, AsyncGenerator
from shared.api.models.retrieval.responses import RAGResponse, CombinedSearchResponse

class LLMHandler:
    
    def __init__(self, client: R2RAsyncClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        load_dotenv()    
        self._chat_model = os.getenv("OLLAMA_CHAT_MODEL")
        self._embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
        self._similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD"))
        self._max_relevant_messages = int(os.getenv("MAX_RELEVANT_HISTORY_MESSAGES"))
        self._ollama_helper = OllamaHelper(
            chat_model=self._chat_model,
            embedding_model=self._embedding_model,
            similarity_threshold=self._similarity_threshold,
            max_relevant_messages=self._max_relevant_messages
        )
        self._max_chat_history_size = int(os.getenv("MAX_CHAT_HISTORY_SIZE"))
        self._chat_history = ChatHistory(max_size=self._max_chat_history_size)
        
    async def search(self, query: str, filters: Optional[dict] = None, limit: Optional[int] = 10) -> CombinedSearchResponse:
        try:
            if filters is None:
                filters = {}
                
            search_settings = {
                "filters": filters,
                "limit": limit,
                "include_scores": True,
                "chunk_settings": {
                    "index_measure": "cosine_distance",
                    "enabled": True
                }
            }
            similar_data = await self._client.retrieval.search(
                query=query,
                search_settings=search_settings
            )
            filtered_data_by_similarity = self._filter_by_similarity(similar_data)
            return filtered_data_by_similarity
        except R2RException as r2re:
            err_msg = f'[-] Error while searching: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while searching: {e} [-]')
            raise Exception(str(e)) from e
        
    def _filter_by_similarity(self, data: List[dict]) -> dict:
        filtered_chunks = [chunk for chunk in data['results']['chunk_search_results'] if chunk['score'] >= self._similarity_threshold]        
        return filtered_chunks
    
    def _create_message(self, role: str, text: str) -> Message:
        """
        Create a new message with computed embedding.
        
        Args:
            role: Role of the message sender ('user' or 'assistant')
            tetx: Content of the message
            
        Returns:
            Message object with computed embedding
        """
        embedding = self._ollama_helper.compute_embedding(text=text)
        return Message(
            role=role, 
            content=text, 
            embedding=embedding
        )
    
    def _get_enhanced_query(self, query: str) -> str:
        """
        Enhance the query with relevant historical context.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query with relevant context
        """
        newest_message = self._create_message(
            role='user', 
            text=query
        )
        
        self._chat_history.add_message(newest_message) 
        
        relevant_messages = self._ollama_helper.get_relevant_messages(
            newest_message.embedding, 
            history=self._chat_history.get_all_messages()
        )
        
        if not relevant_messages:
            return query
            
        # Construct prompt for summarizing relevant history
        history_summary_prompt = self._ollama_helper.construct_history_summary_prompt(
            query=query,
            relevant_history=relevant_messages
        )
        
        # Get summary of relevant context
        context_summary = self._ollama_helper.summarize_context(history_summary_prompt)
        
        # Enhance query with context
        return self._ollama_helper.enhance_user_query(query, context_summary)
    
    def clear_chat_history(self):
        self._chat_history.clear()    
        
    async def async_rag(self, query: str, temperature: int, top_p: int, max_tokens_to_sample: int, limit: Optional[int] = 10) -> AsyncGenerator[RAGResponse, None]:
        try:
            enhanced_query = self._get_enhanced_query(query)        
            return await self._rag_request(
                enhanced_query, 
                temperature, 
                top_p, 
                max_tokens_to_sample, 
                limit, 
                stream=True
            )
        except R2RException as r2re:
            err_msg = f'[-] Error while searching: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while searching: {e} [-]')
            raise Exception(str(e)) from e

    async def rag(self, query: str, temperature: int, top_p: int, max_tokens_to_sample: int, limit: Optional[int] = 10) -> RAGResponse:
        try:
            enhanced_query = self._get_enhanced_query(query)  
            rag_response = await self._rag_request(
                enhanced_query, 
                temperature, 
                top_p, 
                max_tokens_to_sample, 
                limit, 
                stream=False
            )
            return rag_response['results']['completion']['choices'][0]['message']['content']
        except R2RException as r2re:
            err_msg = f'[-] Error while searching: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while searching: {e} [-]')
            raise Exception(str(e)) from e
        
    async def _rag_request(self, query: str, temperature: int, top_p: int, max_tokens_to_sample: int, limit: Optional[int] = 10, stream: bool = False):
        search_settings = {
            "limit": limit,
            "include_scores": True,
            "chunk_settings": {
                "index_measure": "cosine_distance",
                "enabled": True
            }
        }

        rag_generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens_to_sample": max_tokens_to_sample,
        }

        if stream:
            rag_generation_config["stream"] = True

        return await self._client.retrieval.rag(
            query=query,
            rag_generation_config=rag_generation_config,
            search_settings=search_settings
        )    