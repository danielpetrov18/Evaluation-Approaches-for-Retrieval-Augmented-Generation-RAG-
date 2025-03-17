"""Abstracts away the interaction with the underlying LLM."""

# pylint: disable=W0718

import streamlit as st
from streamlit.errors import Error
from r2r import R2RClient, R2RException
from shared.api.models.management.responses import MessageResponse

def retrieve_conversation(client: R2RClient, conversation_id: str) -> MessageResponse | None:
    """Make sure that the conversation exists and return it."""
    try:
        conversation = client.conversations.retrieve(conversation_id).results
        return conversation
    except R2RException as r2re:
        st.error(f"Invalid conversation: {str(r2re)}")
        return None
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")
        return None

def submit_query(client: R2RClient, query: str, conversation_id: str = None) -> str:
    """
    Submit a query to the language model (LLM) and retrieve the response.
    If a new conversation is initiated, the conversation ID is stored in the session state.
    Otherwise we just continue with the existing conversation.

    Args:
        client (R2RClient): The client used to interact with the LLM.
        query (str): The user's query to be submitted.
        conversation_id (str, optional): The ID of the conversation. If not provided, 
            a new conversation is initiated.

    Returns:
        str: The content of the first message in the response from the LLM.
    
    Raises:
        R2RException: If there's an error related to the R2R system.
        Error: For unexpected Streamlit errors.
        Exception: For any other unexpected errors.
    """

    try:
        message = {
            "role": "user",
            "content": query
        }

        search_settings = {
            "use_semantic_search": True,
            "limit": 10,
            "offset": 0,
            "include_metadatas": False,
            "include_scores": True,
            "search_strategy": "vanilla",
        }

        response = client.retrieval.agent(
            message = message,
            search_mode = "custom",
            search_settings = search_settings,
            include_title_if_available = True,
            conversation_id = conversation_id,
        ).results

        # If we are initiating a new conversation
        if not conversation_id:
            st.session_state['conversation_id'] = response.conversation_id

        return response.messages[0].content

    except R2RException as r2re:
        st.error(f"An error while querying LLM: {str(r2re)}")
        #raise r2re
    except Error as e:
        st.error(f"An error occurred: {str(e)}")
        #raise e
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        #raise e
