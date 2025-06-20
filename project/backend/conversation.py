# pylint: disable=C0114
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=C0303
# pylint: disable=E0601
# pylint: disable=R0914
# pylint: disable=W0612
# pylint: disable=W0718

from typing import List, Dict

import requests
import streamlit as st

def list_conversations():
    response: requests.Response = requests.get(
        url="http://r2r:7272/v3/conversations",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        params={
            "offset": 0,
            "limit": 1000
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Error checking conversations: {response.text}")
        return

    conversations: List[Dict] = response.json()["results"]

    if not conversations:
        st.info("No conversations found.")
        return

    for i, conversation in enumerate(conversations, 1):
        with st.expander(
            label=f"Conversation {i}: {conversation['id']}",
            expanded=False
        ):
            st.markdown(f"""
    **ID**: `{conversation['id']}`  
    **Created at**: `{conversation['created_at']}`  
            """, unsafe_allow_html=False)

            with st.popover(label="Delete conversation", icon="🗑️"):
                delete_conv_btn = st.button(
                    label="Confirm",
                    key=f"delete_btn_{conversation['id']}",
                    on_click=delete_conversation,
                    args=(conversation['id'], )
                )
        
    st.info("You've reached the end of the conversations list.")

def delete_conversation(conversation_id: str):
    response: requests.Response = requests.delete(
        url=f"http://r2r:7272/v3/conversations/{conversation_id}",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Failed to delete conversation: {response.status_code} - {response.text}")
        return        

    st.success(f"Deleted conversation: {conversation_id}")

def fetch_messages(conversation_id: str):
    response: requests.Response = requests.get(
        url=f"http://r2r:7272/v3/conversations/{conversation_id}",
        headers={
            "Authorization": f"Bearer {st.session_state['bearer_token']}"
        },
        timeout=5
    )

    if response.status_code != 200:
        st.error(f"Failed to fetch messages: {response.status_code} - {response.text}")
        return

    messages: List[Dict] = response.json()['results']

    if not messages:
        st.info(f"No messages found for conversation: {conversation_id}")
        return

    for i, obj in enumerate(messages, 1):
        message: Dict = obj['message']
        role: str = message['role'].upper()
        content = message['content']
        st.markdown(
            f"**[Message {i}]<br>Role:** `{role}`<br>**Content:** {content}<br>",
            unsafe_allow_html=True
        )
    st.info("You've reached the end of the messages list.")
