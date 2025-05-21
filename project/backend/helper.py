# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=W0718
# pylint: disable=W0719
# pylint: disable=R0903

from typing import List, Dict

import ollama
import numpy as np
import streamlit as st
from .message import Message

@st.cache_resource
def ollama_client():
    """
    This client will be required when interacting with the Ollama server.

    In this project I use the Ollama-client in two different modules:
        - In the `chat` for creating embeddings
        - In the `storage` for invoking a tool

    If running the project locally the hostname would be `localhost`.
    Since this is a containerized application, the hostname should be `host.docker.internal`.
    To modify that behaviour, modify the variable `OLLAMA_API_BASE` in the `env/rag.env` file.
    """
    return ollama.Client(host=st.session_state['ollama_api_base'])

@st.cache_resource
def ollama_options():
    """Options that are going to be used by the Ollama client."""
    return ollama.Options(
        temperature=st.session_state['temperature'],
        top_p=st.session_state['top_p'],
        top_k=st.session_state['top_k'],
        num_ctx=st.session_state["context_window_size"],
        format="json", # This should be json to enforce proper output if required
    )

def get_enhanced_query(query: str) -> str:
    """
    Enhance the query with relevant historical context:
        1. Find messages in the history that are relevant to the given query
        2. Construct prompt for summarizing relevant history
        3. Get summary of relevant context
        4. Enhance query with context
    
    Args:
        query: Original user query
        
    Returns:
        Enhanced query with relevant context
    """
    relevant_messages: List[Dict] = get_relevant_messages(
        query_embedding = st.session_state.messages[-1].embedding,
        history = st.session_state['messages'][:-1] # Get all messages except the last
    )

    if not relevant_messages:
        return query

    history_summary_prompt: str = construct_history_summary_prompt(
        query,
        relevant_messages
    )

    context_summary: str = summarize_context(history_summary_prompt)

    return enhance_user_query(query, context_summary)

@st.cache_data(ttl=None)
def compute_embedding(text: str) -> List[float]:
    """
    Compute the vector embedding of a given text using the selected Ollama model.
    By default this project uses `mxbai-embed-large`, however one can change that from
    the environment variables file under `env/rag.env`.
    
    Args:
        text: Text to compute the embedding for the text input
    
    Raises:
        Exception: If there is an error computing the embedding
    
    Returns:
        List of floats representing the vector embedding of the given text
    """
    try:
        result: Dict = ollama_client().embeddings(
            model=st.session_state['embedding_model'],
            prompt=text,
            options=ollama_options()
        )
        return result['embedding']
    except ollama.ResponseError  as oe:
        st.error(f"Unexpected error computing embeddings: {oe}")
        raise Exception(str(oe)) from oe

@st.cache_data(ttl=None)
def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
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

    vec1: np.ndarray = np.array(embedding1)
    vec2: np.ndarray = np.array(embedding2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_relevant_messages(query_embedding: List[float], history: List[Message]) -> List[Dict]:
    """
    Find messages in the history that are relevant to the given query.
    
    This method iterates over all the provided messages in the history.
    For each message, it computes the cosine similarity between the query embedding.
    If the given message has a similarity exceeding the threshold, it is considered relevant.
    The `relevant` messages are then sorted in descending order, based on their similarity scores.
    Finally, we make sure that only the `top-k` messages are kept.
    
    Args:
        query_embedding: Embedding of the current users query
        history: List of messages to search through
    
    Returns:
        List of messages that are relevant to the query
    """
    if len(history) == 0:
        return []

    relevant_messages: List[Dict] = []
    for message in history:
        similarity: float = compute_similarity(
            embedding1 = query_embedding,
            embedding2 = message.embedding
        )
        if similarity >= st.session_state['similarity_threshold']:
            relevant_messages.append({
                'message': message,
                'similarity': similarity
            })

    # Sort messages based on similarity in descending order (most similar ones are relevant)
    relevant_messages.sort(key=lambda x: x['similarity'], reverse=True)

    # If there're more than n relevant messages, keep only n.
    if len(relevant_messages) > st.session_state['max_relevant_messages']:
        relevant_messages = relevant_messages[:st.session_state['max_relevant_messages']]

    return relevant_messages

def construct_history_summary_prompt(query: str, relevant_history: List[Dict]) -> str:
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
    prompt_parts: List[str] = []
    if relevant_history:
        prompt_parts.append("You are helping to summarize relevant parts of a conversation history that relate to the current question.")
        prompt_parts.append("\n## Relevant conversation history:")

        for i, item in enumerate(relevant_history, 1):
            content: str = item['message'].content
            role: str = item['message'].role
            prompt_parts.append(
                f"{i}. [{role.upper()}]:\n{content}\n\n"
            )

        prompt_parts.append(
            """## Instructions:
1. Extract key information from the conversation history above.
2. Only include details relevant to answering the current question.
3. Keep your summary concise (3-5 sentences maximum).
4. DO NOT add any further explanations or clarifications, just the summary.
"""
        )

    prompt_parts.append(f"## Current question:\n{query}\n")

    if relevant_history:
        prompt_parts.append("## Your summary of the relevant context (be concise):")

    result: str = "\n".join(prompt_parts)
    return result

def summarize_context(history_summary_prompt: str) -> str:
    """
    Generate a summary of relevant historical context using Ollama.
    
    Args:
        augmented_prompt: Previous history + current question of the user
        
    Returns:
        str: Summarized context
    """
    result: Dict = ollama_client().generate(
        model=st.session_state['chat_model'],
        prompt=history_summary_prompt,
        options=ollama_options()
    )

    return result['response']

def enhance_user_query(query: str, context_summary: str) -> str:
    """
    Enhance the current user query with relevant historical context.
    
    Args:
        query: Current user query
        context_summary: Summary of relevant historical context
        
    Returns:
        str: Enhanced query with relevant context
    """
    enhanced: str = f"Relevant conversation context:\n{context_summary}\n\nCurrent query:\n{query}"
    return enhanced
