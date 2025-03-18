"""Abstracts away the interaction with the underlying LLM."""

# pylint: disable=C0301
# pylint: disable=W0718
# pylint: disable=W0719
# pylint: disable=R0903

import json
from uuid import UUID, uuid4
from datetime import datetime
from typing import  Generator
import ollama
import numpy as np
import streamlit as st
from streamlit.errors import Error
from pydantic import BaseModel, Field
from r2r import R2RClient, R2RException

class Message(BaseModel):
    """
    Represents a chat message with metadata and embedding information.
    
    Attributes:
        id: Unique identifier for the message
        role: Role of the message sender (e.g., 'user', 'assistant')
        content: The actual message content
        embedding: Vector embedding of fixed length (1024 dimensions if using the mxbai-embed-large model)
        timestamp: UTC timestamp of when the message was created
    """
    id: UUID = Field(default_factory=uuid4)
    role: str = Field(..., min_length=1)  # ... means required/not nullable
    content: str = Field(..., min_length=1)  # min_length=1 ensures non-empty string
    embedding: list[float] = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # https://docs.pydantic.dev/1.10/usage/model_config/
    class Config:
        """Configuration for the message. A message cannot be modified once created."""
        frozen = True  # Makes instances immutable
        extra = "forbid" # Prevents additional fields not defined in model
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# When using an underscore the client won't be part of the arguments passed to the function
# which are of importance for the caching behaviour.
# https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data
@st.cache_data(ttl=60)  # Cache for 60 seconds
def retrieve_conversation(_client: R2RClient, conversation_id: str) -> list[Message] | None:
    """Make sure that the conversation exists and return it."""
    try:
        conversation = _client.conversations.retrieve(conversation_id).results
        msgs = [
            Message(
                id = obj.id,
                role = obj.message.role,
                content = obj.message.content,
                embedding = obj.metadata['embedding'],
                timestamp = obj.metadata['timestamp']
            )
            for obj in conversation
        ]
        return msgs
    except R2RException as r2re:
        st.error(f"Invalid conversation: {str(r2re)}")
        return None
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")
        return None

def check_conversation_exists(client: R2RClient):
    """Making sure that a conversation exists. If not we create one."""
    try:
        # If the id is empty -> create a new one and use it
        if not st.session_state['conversation_id']:
            st.session_state['conversation_id'] = client.conversations.create().results.id
            st.session_state.messages = []
        else:
            # If the provided id is not related to a conversation raise error
            if not client.conversations.list(ids = [st.session_state['conversation_id']]).results:
                raise R2RException(f"Conversation: {st.session_state['conversation_id']} doesn't exist!", 404)
    except R2RException as r2re:
        st.error(str(r2re))
    except Error as e:
        st.error(str(e))
    except Exception as exc:
        st.error(str(exc))

def add_message(client: R2RClient, msg: dict):
    """Adding a new message to a conversation."""
    try:
        embedding = _compute_embedding(msg['content'])
        new_msg = Message(
            content = msg['content'],
            role = msg['role'],
            embedding = embedding
        )

        parent_id = None
        if st.session_state.messages:
            parent_id = st.session_state.messages[-1].id
            print(parent_id)

        client.conversations.add_message(
            id = st.session_state['conversation_id'],
            content = new_msg.content,
            role = new_msg.role,
            metadata = {
                # Both need to be strings since R2R accepts only strings
                "timestamp": str(new_msg.timestamp),
                "embedding": json.dumps(new_msg.embedding)
            },
            parent_id = str(parent_id) if parent_id else None
        )

        # Then to session state
        if not st.session_state.messages:
            st.session_state.messages = [new_msg]
        else:
            st.session_state.messages.append(new_msg)

    except R2RException as r2re:
        st.error(str(r2re))
        raise R2RException(str(r2re), r2re.status_code) from r2re
    except Exception as e:
        st.error(str(e))
        raise Exception from e

def submit_query(client: R2RClient) -> Generator:
    """
    Submit a query to the language model (LLM) and retrieve the response.
    If a new conversation is initiated, the conversation ID is stored in the session state.
    Otherwise we just continue with the existing conversation.

    Args:
        client (R2RClient): The client used to interact with the LLM.

    Returns:
        str: The content of the first message in the response from the LLM.
    
    Raises:
        R2RException: If there's an error related to the R2R system.
        Error: For unexpected Streamlit errors.
        Exception: For any other unexpected errors.
    """
    try:
        search_settings = {
            "use_semantic_search": True,
            "limit": 10,
            "offset": 0,
            "include_metadatas": False,
            "include_scores": True,
            "search_strategy": "vanilla",
        }

        # Augmented query where I try to store context/relevant history
        enhanced_query = _get_enhanced_query(st.session_state.messages[-1].content)

        generator = client.retrieval.rag(
            query = enhanced_query,
            search_mode = "custom",
            search_settings = search_settings,
            rag_generation_config = {
                "temperature": st.session_state['rag_generation_config']['temperature'],
                "top_p": st.session_state['rag_generation_config']['top_p'],
                "max_tokens_to_sample": st.session_state['rag_generation_config']['max_tokens_to_sample'],
                "stream": True # You must specify this explicitly. In config file it doesn't run.
            }
        )

        return generator

    except R2RException as r2re:
        st.error(f"An error while querying LLM: {str(r2re)}")
        raise r2re
    except Error as e:
        st.error(f"An error occurred: {str(e)}")
        raise e
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e

def extract_completion(generator: Generator) -> Generator:
    """
    Extracts and yields only the text inside <completion>...</completion> tags from a generator.

    Args:
        generator (Generator[str, None, None]): A generator yielding text chunks.

    Yields:
        str: Extracted text within <completion> tags.
     """

    inside_completion = False  # Track whether we are inside the <completion> section

    for chunk in generator:
        if "<completion>" in chunk:
            inside_completion = True
            after_tag = chunk.split("<completion>", 1)[1]

            # Handle cases where closing tag is in the same chunk
            if "</completion>" in after_tag:
                before_close, _ = after_tag.split("</completion>", 1)
                yield before_close.strip()
                inside_completion = False
            else:
                yield after_tag.strip()
            continue

        if inside_completion:
            if "</completion>" in chunk:
                before_close, _ = chunk.split("</completion>", 1)
                yield before_close.strip()
                inside_completion = False
            else:
                yield chunk.strip()

def _get_enhanced_query(query: str) -> str:
    """
    Enhance the query with relevant historical context.
    
    Args:
        query: Original user query
        
    Returns:
        Enhanced query with relevant context
    """
    # Take all assistant messages which are relevant for context
    assistant_messages = [msg for msg in st.session_state.messages if msg.role == "assistant"]

    # If json.loads() makes no sense refer to add_message()
    relevant_messages = _get_relevant_messages(
        query_embedding = st.session_state.messages[-1].embedding,
        history = assistant_messages
    )

    if not relevant_messages:
        return query

    # Construct prompt for summarizing relevant history
    history_summary_prompt = _construct_history_summary_prompt(query, relevant_messages)

    # Get summary of relevant context
    context_summary = _summarize_context(history_summary_prompt)

    # Enhance query with context
    return _enhance_user_query(query, context_summary)

def _compute_embedding(text: str) -> list[float]:
    """
    Compute the vector embedding of a given text using the selected Ollama model.
    
    Args:
        text: Text to compute the embedding for the text input
    
    Raises:
        ollama.ResponseError: If there is an error in the Ollama response
        Exception: If there is an error computing the embedding
    
    Returns:
        List of floats representing the vector embedding of the given text
    """
    try:
        # Since this function can be used for batch processing, it returns a list of embeddings.
        # We only one the first one.
        response = ollama.embed(
            model = st.session_state['embedding_model'],
            input = text
        )
        return response['embeddings'][0]
    except ollama.ResponseError as oe:
        st.error(f"Ollama error computing embeddings: {oe.error}")
        raise ollama.ResponseError(oe.error) from oe
    except Exception as e:
        st.error(f"Unexpected error computing embeddings: {e}")
        raise Exception(str(e)) from e

def _compute_similarity(embedding1: list[float], embedding2: list[float]) -> float:
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

    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def _get_relevant_messages(query_embedding: list[float], history: list[Message]) -> list[dict]:
    """
    Find messages in the history that are relevant to the given query.
    I assume only assistant messages can be relevant since they are generated by the model.
    
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
        similarity = _compute_similarity(
            embedding1 = query_embedding,
            embedding2 = json.loads(message.embedding)
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

def _construct_history_summary_prompt(query: str, relevant_history: list[dict]) -> str:
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
            content = item['message'].content
            role = item['message'].role
            similarity = item['similarity']
            prompt_parts.append(
                f"{i+1}. [{role} message (similarity: {similarity:.2f})]: {content}"
            )
        prompt_parts.append(
            "\nPlease summarize the history context concisely which is relevant to the current question:"
        )

    prompt_parts.append(f"Current question: {query}")

    result = "\n".join(prompt_parts)
    return result

def _summarize_context(history_summary_prompt: str) -> str:
    """
    Generate a summary of relevant historical context using Ollama.
    
    Args:
        augmented_prompt: Previous history + current question of the user
        
    Returns:
        str: Summarized context
    """
    resp = ollama.generate(st.session_state['chat_model'], history_summary_prompt)['response']
    return resp

def _enhance_user_query(query: str, context_summary: str) -> str:
    """
    Enhance the current user query with relevant historical context.
    
    Args:
        query: Current user query
        context_summary: Summary of relevant historical context
        
    Returns:
        str: Enhanced query with relevant context
    """
    enhanced = f"Relevant conversation context:\n{context_summary}\n\nCurrent query:\n{query}"
    return enhanced
