# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=W0718
# pylint: disable=W0719
# pylint: disable=R0903

import json
from typing import Generator, Union, List, Dict

import streamlit as st
from streamlit.errors import Error
from r2r import R2RClient, R2RException
from shared.api.models.retrieval.responses import SSEEventBase, MessageEvent

from .message import Message
from .helper import (
    compute_embedding,
    get_enhanced_query
)

@st.cache_data(ttl=60)  # Cache for 60 seconds
def retrieve_messages(_client: R2RClient, conversation_id: str) -> Union[List[Message],None]:
    """
                          ^
                          |
                          |
    When using an underscore the client won't be part of the arguments passed to the function
    which are of importance for the caching behaviour. Since client cannot be pickled/serialized.
    https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data
    
    Make sure that the conversation exists and return it.
    This method also makes sure that R2R messages are converted into my own Message class.
    """
    try:
        conversation = _client.conversations.retrieve(conversation_id).results
        msgs = [
            Message(
                id = str(obj.id),
                role = obj.message.role,
                content = obj.message.content,
                # Make sure you convert it back to an embedding -> check add_message()
                embedding = json.loads(obj.metadata['embedding']),
            )
            for obj in conversation
        ]
        return msgs
    except R2RException as r2re:
        st.error(f"Invalid conversation: {r2re.message}")
        return None
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")
        return None

def check_conversation_exists(client: R2RClient):
    """
    Making sure that a conversation exists. If not we create one.
    
    If the id is empty -> create a new one and use it
        This is the case when the user starts a new conversation
    """
    try:
        if not st.session_state['conversation_id']:
            st.session_state['conversation_id'] = client.conversations.create().results.id
            st.session_state.messages = []
            st.session_state['parent_id'] = None
        else:
            # If the provided id is not related to a conversation raise error
            if not client.conversations.list(ids = [st.session_state['conversation_id']]).results:
                raise R2RException(
                    message=f"Conversation: {st.session_state['conversation_id']} doesn't exist!",
                    status_code=404
                )
    except R2RException as r2re:
        st.error(r2re.message)
    except Error as e:
        st.error(str(e))
    except Exception as exc:
        st.error(str(exc))

def set_new_prompt(client: R2RClient, prompt_name: str) -> bool:
    """After making sure the prompt exists we select it for future RAG completions"""
    try:
        prompt_obj = client.prompts.retrieve(prompt_name).results
        st.session_state['selected_prompt'] = prompt_name
        st.session_state['prompt_template'] = prompt_obj.template
        return True
    except R2RException as r2re:
        st.error(r2re.message)
        return False
    except Exception as e:
        st.error(str(e))
        return False

def add_message(client: R2RClient, msg: Dict[str, str]):
    """
    Adding a new message to a conversation.
    
    For each added message an embedding is computed.
    It is then used when quering for relevant messages from previous interactions.
    """
    try:
        embedding = compute_embedding(msg['content'])

        msg_response = client.conversations.add_message(
            id = st.session_state['conversation_id'],
            content = msg['content'],
            role = msg['role'],
            metadata = {
                # Needs to be converted to a string, since R2R cannot accept a list[float]
                "embedding": json.dumps(embedding)
            },
            # If this is the first message in the conversation => None/Null
            parent_id = st.session_state['parent_id']
        )

        # Set the parent id for next message to equal the id of the newly added one
        st.session_state['parent_id'] = str(msg_response.results.id)

        new_msg = Message(
            id = st.session_state['parent_id'],
            content = msg['content'],
            role = msg['role'],
            embedding = embedding
        )

        # Finally, add to session state to be displayed
        if not st.session_state.messages:
            st.session_state.messages = [new_msg]
        else:
            st.session_state.messages.append(new_msg)

    except R2RException as r2re:
        st.error(r2re.message)
        raise R2RException(str(r2re), r2re.status_code) from r2re
    except Exception as e:
        st.error(str(e))
        raise Exception from e

def submit_query(client: R2RClient) -> Generator[SSEEventBase, None, None]:
    """
    * This method is first going to select relevant messages as a history.
    * Those will be used to augment the query and then the `LLM` will be asked to generate a response.
    * R2R will then retrieve relevant chunks using semantic similarity.
    * The retrieved chunks will be re-ranked.
    * Finally all the chunks will be used to generate the final response.

    Args:
        client (R2RClient): The client used to interact with the LLM.

    Returns:
        Generator: Response from the LLM as a generator.
    
    Raises:
        R2RException: If there's an error related to the R2R system.
        Error: For unexpected Streamlit errors.
        Exception: For any other unexpected errors.
    """
    try:
        search_settings: dict = {
            "use_semantic_search": True,
            "limit": st.session_state['top_k'],
            "offset": 0,
            "include_metadatas": False,
            "include_scores": True,
            "search_strategy": "vanilla",
        }

        # Augmented query which should include additional data if any is found
        enhanced_query: str = get_enhanced_query(st.session_state.messages[-1].content)

        generator: Generator[SSEEventBase, None, None] = client.retrieval.rag(
            query = enhanced_query,
            rag_generation_config = {
                "model": f"ollama_chat/{st.session_state['chat_model']}",
                "temperature": st.session_state['temperature'],
                "top_p": st.session_state['top_p'],
                "max_tokens_to_sample": st.session_state['max_tokens'],
                "stream": True # You must specify this explicitly. In config file it doesn't run.
            },
            search_mode = "custom",
            search_settings = search_settings,
            task_prompt=st.session_state['prompt_template'],
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

def extract_completion(generator: Generator[SSEEventBase, None, None]) -> Generator:
    """
    Extracts and yields only the text inside <completion>...</completion> tags from a generator
    while preserving original formatting.
    
    Args:
        generator (Generator[SSEEventBase, None, None]): 
            A generator yielding various events.
        
    Yields:
        str: Only data that is generated on the fly relevant for the final answer.
    """
    for event in generator:
        if isinstance(event, MessageEvent):
            yield event.data.delta.content[0].payload.value
