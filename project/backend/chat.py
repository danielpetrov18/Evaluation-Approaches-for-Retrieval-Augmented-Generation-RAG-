# pylint: disable=C0114
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=W0718
# pylint: disable=W0719
# pylint: disable=R0903

from typing import Union, List, Dict, Final, Any

import requests
import streamlit as st

# https://r2r-docs.sciphi.ai/api-and-sdks/retrieval/search-app
SEARCH_SETTINGS: Final[Dict[str, Any]] = {
    "use_semantic_search": True,
    "limit": st.session_state['top_k'],
    "offset": 0,
    "include_metadatas": False,
    "include_scores": True,
    "search_strategy": "vanilla",
    "chunk_settings": {
        "index_measure": "cosine_distance",
        "enabled": True,
        "ef_search": 80
    }
}

# https://r2r-docs.sciphi.ai/api-and-sdks/retrieval/rag-app
RAG_GENERATION_CONFIG: Final[Dict[str, Any]] = {
    "model": f"ollama_chat/{st.session_state['chat_model']}",
    "temperature": st.session_state['temperature'],
    "top_p": st.session_state['top_p'],
    "max_tokens_to_sample": st.session_state['max_tokens'],
    "stream": False
}

def retrieve_messages(conversation_id: str) -> Union[List[Dict[str, str]], None]:
    response: requests.Response = requests.get(
        url=f"http://r2r:7272/v3/conversations/{conversation_id}",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Failed to fetch messages: {response.status_code} - {response.text}")
        return None

    conversation: List[Dict] = response.json()['results']
    messages: List[Dict[str, str]] = [
        {
            "id": obj["id"],
            "role": obj["message"]["role"],
            "content": obj["message"]["content"]
        }
        for obj in conversation
    ]
    return messages

def check_conversation_exists() -> bool:
    response: requests.Response = requests.get(
        url=f"http://r2r:7272/v3/conversations/{st.session_state['conversation_id']}",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        params={
            "limit": 1000
        },
        timeout=5
    )

    if response.status_code != 200:
        return False

    return True

def create_conversation():
    response: requests.Response = requests.post(
        url="http://r2r:7272/v3/conversations",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Failed to create conversation: {response.status_code} - {response.text}")
        return

    st.session_state['conversation_id'] = response.json()['results']['id']
    st.session_state['messages'] = []
    st.session_state['parent_id'] = None

def set_new_prompt(prompt_name: str) -> bool:
    response: requests.Response = requests.post(
        url=f"http://r2r:7272/v3/prompts/{prompt_name}",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        timeout=5
    )

    if response.status_code != 200:
        return False

    st.session_state['selected_prompt'] = prompt_name
    st.session_state['prompt_template'] = response.json()['results']['template']
    return True

def add_message(msg: Dict[str, str]):
    response: requests.Response = requests.post(
        url=f"http://r2r:7272/v3/conversations/{st.session_state['conversation_id']}/messages",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        json={
            "content": msg['content'],
            "role": msg['role'],
            # If this is the first message in the conversation => None/Null
            "parent_id": st.session_state['parent_id'] 
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Failed to add message: {response.status_code} - {response.text}")
        return

    # Set the parent id for next message to equal the id of the newly added one
    st.session_state['parent_id'] = response.json()['results']['id']

    # Finally, add to session state to be displayed
    if not st.session_state['messages']:
        st.session_state['messages'] = [msg]
    else:
        st.session_state['messages'].append(msg)

def submit_query() -> str:
    query: str = st.session_state['messages'][-1]['content']

    # 1. Request and retrieve the context
    response: requests.Response = requests.post(
        url="http://r2r:7272/v3/retrieval/search",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        json={
            "query": query,
            "search_settings": SEARCH_SETTINGS,
            "search_mode": "custom"
        },
        timeout=60
    )

    if response.status_code != 200:
        st.error(f"Failed to retrieve context: {response.status_code} - {response.text}")
        return None

    # Extract the relevant context (if any)
    # Currently this is very naive - no context filtering and no notion of - is the source relevant
    # One could use a LLM call to classify them
    retrieved_chunks = []
    for chunk in response.json()['results']['chunk_search_results']:
        if chunk['text']:
            retrieved_chunks.append(chunk['text'])

    # 2. Augment the user query with the context
    user_msg: str = st.session_state['prompt_template'].format(
        context="\n".join(retrieved_chunks),
        query=query
    )

    messages: List[Dict] = st.session_state['messages'][:-1] # Exclude the query
    messages.append({'role': 'user', 'content': user_msg})   # This will be the augmented prompt (query + context)

    # 3. Send a RAG request
    response: requests.Response = requests.post(
        url="http://r2r:7272/v3/retrieval/completion",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        json={
            "messages": messages,
            "generation_config": RAG_GENERATION_CONFIG,
            "response_model": "MessageEvent",
        },
        timeout=600 # 10 minutes
    )

    if response.status_code != 200:
        st.error(f"Failed to stream response: {response.status_code} - {response.text}")
        return None

    return response.json()['results']['choices'][0]['message']['content']
