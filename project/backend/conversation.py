# pylint: disable=C0114
# pylint: disable=E0601
# pylint: disable=R0914
# pylint: disable=W0612
# pylint: disable=W0718

import io
import os
import json
import datetime
from pathlib import Path
from typing import List
import requests
import pandas as pd
import streamlit as st
from streamlit.errors import Error
from r2r import R2RClient, R2RException

def list_conversations(
    client: R2RClient,
    ids: List[str] = None,
    offset: int = 0,
    limit: int = 100
):
    """List conversations."""
    try:
        conversations = client.conversations.list(ids, offset, limit).results

        if conversations:
            st.write(f"Showing conversations {offset+1} to {offset+len(conversations)}")

            for i, conversation in enumerate(conversations):
                with st.expander(
                    label=f"Conversation {i + 1}: {conversation.id}",
                    expanded=False
                ):
                    st.json(conversation)

                    with st.popover(label="Delete conversation", icon="🗑️"):
                        delete_conv_btn = st.button(
                            label="Confirm deletion",
                            key=f"delete_conversation_{i}",
                            on_click=delete_conversation,
                            args=(client, conversation.id,)
                        )

            if len(conversations) < limit:
                st.info("You've reached the end of the conversations list.")
        else:
            if st.session_state["page_number"] > 0:
                st.warning("No more conversations found.")
            else:
                st.info("No conversations found")
    except R2RException as r2re:
        st.error(f"Error checking conversations: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def delete_conversation(client: R2RClient, conversation_id: str):
    """Delete a conversation."""
    try:
        client.conversations.delete(conversation_id)
        st.session_state["page_number"] = 0
        st.success(body=f"Deleted conversation: {conversation_id}")
    except R2RException as r2re:
        st.error(f"Error deleting conversation: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def fetch_messages(client: R2RClient, conversation_id: str, display_metadata: bool):
    """Fetch all messages related to a particular conversation."""
    try:
        messages = client.conversations.retrieve(conversation_id).results

        if not messages:
            st.warning(f"No messages found for conversation: {conversation_id}")
        else:
            st.markdown(f"**Showing messages for conversation: {conversation_id}**")

            for i, obj in enumerate(messages):
                with st.expander(
                    label=f"Message {i+1}: [{obj.id}]",
                    expanded=True,
                    icon="📝"
                ):
                    role = obj.message.role.upper()
                    content = obj.message.content
                    st.markdown(
                        f"**Role:** {role}<br>**Content:** {content}<br>",
                        unsafe_allow_html=True
                    )
                    if display_metadata:
                        st.json(obj.metadata)
            st.info("You've reached the end of the messages list.")
    except R2RException as r2re:
        st.error(f"Error fetching messages: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def export_conversations(conversations_ids: str, out: str):
    """Export conversations to exports. (Only metadata)"""
    try:
        # Doesn't work since httpx doesn't have support for context manager.
        # Using the client by 'with' keyword
        # Workaround: Use raw requests
        # client.conversations.export(
        #     output_path = str(out),
        #     columns = [ 'id', 'created_at', 'name'],
        #     filters = filters,
        #     include_header = True
        # )

        headers = {
            'Authorization': f'Bearer {st.session_state['bearer_token']}',
            'Content-Type': 'application/json',
            'Accept': 'text/csv'
        }

        payload = {
            'include_header': 'true',
            'columns': [
                'id',
                'created_at',
                'name'
            ]
        }

        response = requests.post(
            url='http://r2r:7272/v3/conversations/export',
            headers=headers,
            json=payload,
            timeout=5
        )

        if response.status_code != 200:
            raise R2RException(response.text, response.status_code)

        df = pd.read_csv(io.BytesIO(response.content))
        if df.shape[0] == 0: # If the dataframe is empty (no rows)
            raise R2RException('No conversations found', 404)

        filters = {}
        if conversations_ids:
            filters['id'] = [filter_id.strip() for filter_id in conversations_ids.split('\n')]

        # Apply filters in-place
        if filters:
            _filter_df(df, filters)

        curr_path = os.getcwd()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = Path(curr_path) / st.session_state['exports_dir'] / f'{out}_{timestamp}.csv'

        df.to_csv(out, index=False)

        st.success("Conversations exported successfully")
    except R2RException as r2re:
        st.error(f"Error exporting conversations: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def export_messages(out: str, filters: dict = None):
    """Export messages to a CSV file."""
    try:
        payload = {
            "include_header": "true"
        }

        response = requests.post(
            url='http://r2r:7272/v3/conversations/export_messages',
            headers={
                'Authorization': f'Bearer {st.session_state['bearer_token']}',
                "Content-Type": "application/json",
                "Accept": "text/csv"
            },
            json=payload,
            timeout=5
        )

        if response.status_code != 200:
            raise R2RException(response.text, response.status_code)

        df = pd.read_csv(io.BytesIO(response.content))
        if df.shape[0] == 0: # If the dataframe is empty (no rows)
            raise R2RException('No messages found', 404)

        df['content'] = df['content'].apply(json.loads)

        df['role'] = df['content'].apply(lambda x: x.get('role', None))
        df['message'] = df['content'].apply(lambda x: x.get('content', None))

        df = df[
            [
                'id', 
                'conversation_id', 
                'created_at', 
                'parent_id', 
                'role', 
                'message'
            ]
        ]

        if filters:
            if filters.get("role"):
                df = df.loc[df["role"] == filters["role"]]

            if filters.get("min_msg_length"):
                df = df.loc[df["message"].str.len() > filters["min_msg_length"]]

            if filters.get("conversation_id"):
                df = df.loc[df['conversation_id'].isin(filters["conversation_id"])]

        curr_path = os.getcwd()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = Path(curr_path) / st.session_state['exports_dir'] / f'{out}_{timestamp}.csv'

        df.to_csv(out, index=False)

        st.success("Successfully exported messages")

    except R2RException as r2re:
        st.error(f"Error exporting messages: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def _filter_df(df: pd.DataFrame, filters: dict):
    for column, value in filters.items():
        if column in df.columns:
            if isinstance(value, list):  # If filtering with multiple values
                df = df[df[column].isin(value)]
            else:  # Single value filter
                df = df[df[column] == value]
