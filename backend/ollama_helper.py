import os
import ollama
import numpy as np
from message import Message
from typing import List, Dict
from logging import error, info

class OllamaHelper:

    def __init__(self, chat_model: str, embedding_model: str, similarity_threshold: float = 0.5, max_relevant_messages: int = 5):
        """
        This class serves as a helper to enhance the history context of the  rag-based system.
        The idea is to fetch relevant messages based on the current query of the user.
        If relevant messages have been discovered they get ordered by similarity and the 5 most relevant
            messages are used by Ollama to generate a summary of the relevant history.
        
        Args:
            chat_model: Name of the Ollama model to use
            embedding_model: Name of the Ollama model to use for embeddings
            similarity_threshold: Minimum similarity score to consider a message relevant
            max_relevant_messages: Maximum number of relevant messages to return
        """
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._similarity_threshold = similarity_threshold
        self._max_relevant_messages = max_relevant_messages
        self._embedding_model_dimension = int(os.getenv("OLLAMA_EMBEDDING_MODEL_DIMENSION"))

    def compute_embedding(self, text: str) -> List[float]: 
        """
        Compute the vector embedding of a given text using the selected Ollama model.
        By default it uses the 'mxbai-embed-large' embedding model.
        
        Args:
            text: Text to compute the embedding for the text input
        
        Raises:
            ollama.ResponseError: If there is an error in the Ollama response
            Exception: If there is an error computing the embedding
        
        Returns:
            List of floats representing the vector embedding of the given text
        """
        try:
            # Since this function can be used for batch processing, it returns a list of embeddings. We only one the first one.
            return ollama.embed(model=self._embedding_model, input=text)['embeddings'][0]
        except ollama.ResponseError as oe:
            error(f"[-] Ollama error computing embeddings: {oe.error} [-]")
            raise Exception(oe.error)
        except Exception as e:
            error(f"[-] Unexpected error computing embeddings: {e} [-]")
            raise Exception(e)

    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute the cosine similarity between two vector embeddings.
        
        Args:
            embedding1: First vector embedding
            embedding2: Second vector embedding
        
        Returns:
            float: Cosine similarity score between the two embeddings. 
            The closer to 1, the more similar the embeddings are.
        
        Raises:
            ValueError: If the embeddings have different lengths
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same length!")
        
        if len(embedding1) != self._embedding_model_dimension:
            raise ValueError(f"Embeddings must have a length of {self._embedding_model_dimension}! Refer to the ollama documentation for {self._embedding_model}!")
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def get_relevant_messages(self, query_embedding: List[float], history: List[Message]) -> List[Dict]:   
        """
        Find messages in the history that are relevant to the given query.
        
        Args:
            query_embedding: Embedding of the current users query
            history: List of messages to search through
        
        Returns:
            List of messages that are relevant to the query
        """
        if len(history) == 0:
            return []
        
        relevant_messages = []
        for message in history:
            similarity = self._compute_similarity(query_embedding, message.embedding)
            if similarity >= self._similarity_threshold:
                relevant_messages.append({
                    'content': message.content,
                    'similarity': similarity,
                    'role': message.role
                })
        
        # Sort messages based on similarity in descending order (most similar ones are relevant)
        relevant_messages.sort(key=lambda x: x['similarity'], reverse=True)
        
        # If there're more than n relevant messages, keep only n, where n is self._max_relevant_messages
        if len(relevant_messages) > self._max_relevant_messages:
            relevant_messages = relevant_messages[:self._max_relevant_messages]
        
        return relevant_messages
    
    def construct_history_summary_prompt(self, query: str, relevant_history: List[Dict]) -> str:
        """
        Construct an augmented prompt that includes relevant historical context.
        This is going to be used by the Ollama model to generate a summary of the relevant history.
        This summary is then used to enhance the user query and then we can make a RAG-based invocation.
        
        Args:
            query: Current user query
            relevant_history: List of relevant historical messages
            
        Returns:
            Augmented prompt string
        """
        prompt_parts = []
        if relevant_history:
            prompt_parts.append("Relevant conversation history:")
            for i, item in enumerate(relevant_history):
                content = item['content']
                similarity = item['similarity']
                role = "User" if item['role'] == 'user' else "Assistant"
                prompt_parts.append(
                    f"{i+1}. [{role} message (similarity: {similarity:.2f})]: {item['content']}"
                )
            prompt_parts.append("\nPlease summarize the history context concisely which is relevant to the current question:")

        prompt_parts.append(f"Current question: {query}")

        result = "\n".join(prompt_parts)
        return result
    
    def summarize_context(self, history_summary_prompt: str) -> str:
        """
        Generate a summary of relevant historical context using Ollama.
        
        Args:
            augmented_prompt: Previous history + current question of the user
            
        Returns:
            str: Summarized context
        """
        resp = ollama.generate(model=self._chat_model, prompt=history_summary_prompt)['response']
        return resp
    
    def enhance_user_query(self, query: str, context_summary: str) -> str:
        """
        Enhance the current user query with relevant historical context.
        
        Args:
            query: Current user query
            context_summary: Summary of relevant historical context
            
        Returns:
            str: Enhanced query with relevant context
        """
        enhanced = f"Relevant conversation context:\n{context_summary}\n\nCurrent query:\n{query}\n\nEnhanced query:\n"
        return enhanced