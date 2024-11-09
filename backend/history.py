import ollama
import logging
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple

class ChatClient:
    
    def __init__(
        self, 
        chat_model: str, 
        embedding_model: str, 
        max_history: int = 30, 
        similarity_threshold: float = 0.65, 
        max_context_items: int = 5
    ):
        """
        Args:
            chat_model: Model name for chat completion
            embedding_model: Model name for computing embeddings
            max_history: Maximum number of messages to keep in history
            similarity_threshold: Minimum similarity score to consider a message relevant
            max_context_items: Maximum number of relevant history items to include
        """
        self.__chat_model = chat_model
        self.__embedding_model = embedding_model
        self.__max_history = max_history
        self.__similarity_threshold = similarity_threshold
        self.__max_context_items = max_context_items
        self.__message_history: deque = deque(maxlen=max_history)
        self.__logger = logging.getLogger(__name__)
        
        self.__logger.info(f'[+] Chat client created with chat model: {chat_model}, embedding model: {embedding_model}, max history: {max_history}, similarity threshold: {similarity_threshold}, max context items: {max_context_items} [+]')

    def _compute_embeddings(self, text: str) -> List[float]:  
        """
        Compute embeddings for a given text using the embedding model.

        Args:
            text: The text to compute embeddings for

        Returns:
            A list of floats representing the computed embeddings

        Raises:
            ollama.ResponseError: If there is an issue with the response from the embedding model
            Exception: If there is an unexpected issue computing embeddings
        """
        try:
            response = ollama.embed(model=self.__chat_model, input=text)
            return response['embeddings']
        except ollama.ResponseError as oe:
            self.logger.error(f"Error computing embeddings: {oe}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error computing embeddings: {e}")
            raise

    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate the cosine similarity between two embeddings.
        
        https://datastax.medium.com/how-to-implement-cosine-similarity-in-python-505e8ec1d823

        Args:
            embedding1: First list of float numbers representing the first embedding.
            embedding2: Second list of float numbers representing the second embedding.

        Returns:
            A float representing the cosine similarity between the two embeddings.
            If both vectors are identical and have the same direction then the cosine similarity will be 1.
            If both vectors are completely different and perpendicular to each other then the cosine similarity will be 0.
            If both vectors are dissimilar and have the opposite direction then the cosine similarity will be -1.

        Raises:
            ValueError: If the lengths of the two embeddings are not the same.
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same length.")
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _find_relevant_history(self, query: str) -> List[Dict]:
        """
        Find relevant messages from history based on semantic similarity.
        
        Args:
            query: Current user query
            
        Returns:
            List of relevant messages with their similarity scores
        """
        if not self.message_history:
            return []

        query_embedding = self._compute_embeddings(query)
        relevant_messages = []

        for message in self.message_history:
            # Skip computing embeddings for system messages
            if message['role'] == 'system':
                continue

            # Compute similarity with message content
            message_embedding = self._compute_embeddings(message['content'])
            similarity = self._compute_similarity(query_embedding, message_embedding)

            if similarity >= self.similarity_threshold:
                relevant_messages.append({
                    'message': message,
                    'similarity': similarity
                })

        # Sort by similarity and take top k most relevant
        relevant_messages.sort(key=lambda x: x['similarity'], reverse=True)
        return relevant_messages[:self.max_context_items]

    def _construct_augmented_prompt(self, query: str, relevant_history: List[Dict]) -> str:
        """
        Construct an augmented prompt that includes relevant historical context.
        
        Args:
            query: Current user query
            relevant_history: List of relevant historical messages
            
        Returns:
            Augmented prompt string
        """
        prompt_parts = []

        if relevant_history:
            prompt_parts.append("Relevant conversation history:")
            for i, item in enumerate(relevant_history, 1):
                message = item['message']
                similarity = item['similarity']
                role = "Human" if message['role'] == 'user' else "Assistant"
                prompt_parts.append(
                    f"{i}. [{role} message (similarity: {similarity:.2f})]: {message['content']}"
                )
            prompt_parts.append("\nGiven this context, please answer the following question:")

        prompt_parts.append(f"Current question: {query}")
        
        return "\n".join(prompt_parts)

    async def chat(self, query: str, system_prompt: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """
        Process a chat query with semantic search augmentation.
        
        Args:
            query: User's question
            system_prompt: Optional system prompt to guide the model
            
        Returns:
            Tuple of (response text, relevant history items used)
        """
        try:
            # Find relevant historical context
            relevant_history = self._find_relevant_history(query)
            
            # Construct the augmented prompt
            augmented_prompt = self._construct_augmented_prompt(query, relevant_history)
            
            # Prepare messages for the chat
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': augmented_prompt})

            # Get response from Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )

            # Extract the response content
            response_content = response['message']['content']

            # Update message history
            self.message_history.append({'role': 'user', 'content': query})
            self.message_history.append({'role': 'assistant', 'content': response_content})

            return response_content, relevant_history

        except Exception as e:
            self.logger.error(f"Error in chat processing: {e}")
            raise

    def get_message_history(self) -> List[Dict]:
        return list(self.message_history)

    def clear_history(self):
        self.message_history.clear()