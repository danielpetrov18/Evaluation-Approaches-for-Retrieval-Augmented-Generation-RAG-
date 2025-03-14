"""GUI to manage conversations."""

import sys
from uuid import uuid4
from pathlib import Path
from r2r import R2RException
import streamlit as st
from streamlit.errors import Error
from st_app import load_client # pylint: disable=E0401

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from conversations import Conversations  # pylint: disable=C0401

@st.cache_resource
def get_conversations_handler():
    """Get the conversations handler."""
    return Conversations(client=load_client())

def list_conversations(ids: list[str] = None, offset: int = 0, limit: int = 100):
    """List conversations."""
    try:
        conversations_response = load_client().conversations.list(
            ids = ids,
            offset = offset,
            limit = limit
        )
        conversations = conversations_response.results

        if conversations:
            st.write(f"Showing conversations {offset+1} to {offset+len(conversations)}")

            # Display each conversation in a cleaner format
            for i, conversation in enumerate(conversations):
                with st.expander(
                    label=f"Conversation: {conversation.name} ({conversation.id})",
                    expanded=False
                ):
                    st.json(conversation)

                    update_col, delete_col = st.columns(2)

                    with update_col:
                        with st.popover(
                            label="Rename conversation",
                            icon="✏️"
                        ):
                            new_name_key = str(uuid4())
                            new_name = st.text_input( # pylint: disable=W0612
                                label="Rename conversation",
                                placeholder="Enter new name",
                                key=new_name_key
                            )
                            update_conv_btn = st.button( # pylint: disable=W0612
                                label="Update Name",
                                key=f"update_conversation_{i}",
                                on_click=update_conversation,
                                args=(conversation.id, new_name_key)
                            )

                    with delete_col:
                        with st.popover(
                            label="Delete conversation",
                            icon="🗑️",
                        ):
                            delete_conv_btn = st.button( # pylint: disable=W0612
                                label="Confirm deletion",
                                key=f"delete_conversation_{i}",
                                on_click=delete_conversation,
                                args=(conversation.id,)
                            )

            # Show a message if we've reached the end
            if len(conversations) < limit:
                st.info("You've reached the end of the conversations list.")
        else:
            if st.session_state["page_number"] > 0:
                st.warning("No more conversations found.")
            else:
                st.info("No conversations found")
    except R2RException as r2re:
        st.error(f"Error checking conversations: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def fetch_messages(conversation_id: str, display_metadata: bool):
    """Fetch all messages related to a particular conversation."""
    try:
        messages = load_client().conversations.retrieve(conversation_id).results

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
                    if display_metadata:
                        st.markdown(
                            # pylint: disable=C0301
                            f"**Role:** {obj.message.role.upper()}<br>**Content:** {obj.message.content}<br>",
                            unsafe_allow_html=True
                        )
                        st.json(obj.metadata)
                    else:
                        st.markdown(
                            # pylint: disable=C0301
                            f"**Role:** {obj.message.role.upper()}<br>**Content:** {obj.message.content}",
                            unsafe_allow_html=True
                        )

    except R2RException as r2re:
        st.error(f"Error fetching messages: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def delete_conversation(del_conversation_id: str):
    """Delete a conversation."""
    try:
        load_client().conversations.delete(del_conversation_id)
        st.session_state["page_number"] = 0
        st.success(body=f"Deleted conversation: {del_conversation_id}", icon="🗑️")
    except R2RException as r2re:
        st.error(f"Error deleting conversation: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def update_conversation(update_conversation_id: str, input_name_key: str):
    """
    Change the name of an existing conversation.
    
    Since I had problems with this particular functionality I decided to capture
    the new name from the widget since it's automatically stored in the session
    state by Streamlit.
    """
    try:
        new_name = st.session_state[input_name_key]
        update_resp = load_client().conversations.update(
            id=update_conversation_id,
            name=new_name
        ).results
        st.session_state["page_number"] = 0
        st.success(
            body=f"Updated name of {update_conversation_id} to {update_resp.name}!",
            icon="✅"
        )
    except R2RException as r2re:
        st.error(f"Error updating conversation: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def export_conversations(conversations_ids: str, output_path: str):
    """Export conversations to /backend/exports. (Only metadata)"""
    try:
        filters = {}
        if conversations_ids:
            filters['id'] = [filter_id.strip() for filter_id in conversations_ids.split('\n')]

        get_conversations_handler().export_conversations_to_csv(
            bearer_token = st.session_state["token"],
            out = output_path,
            filters = filters
        )
        st.success("Conversations exported successfully")
    except R2RException as r2re:
        st.error(f"Error exporting conversations: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def export_messages(outpath: str):
    """Exports all messages of all conversations."""
    try:
        get_conversations_handler().export_messages_to_csv(
            bearer_token = st.session_state["token"],
            out = outpath
        )
        st.success("Messages exported successfully")
    except R2RException as r2re:
        st.error(f"Error exporting messages: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

if __name__ == "__page__":
    st.title("🗪 Manage conversations")

    if "items_per_page" not in st.session_state:
        st.session_state["items_per_page"] = 10

    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 0

    t_list, t_conv_msgs, t_exp_conversations, t_exp_msgs  = st.tabs(
        [
            "List Conversations",
            "Conversation messages",
            "Export Conversations",
            "Export Messages"
        ]
    )

    with t_list:
        st.markdown("**List Conversations**")

        filter_ids = st.text_area(
            label="Filter by Conversation IDs",
            placeholder="id1\nid2\n... (optional)",
            value="",
            help="Enter conversation ids on each line",
            height=100
        )

        # Pagination controls
        col1, col2 = st.columns(spec=[3, 1])
        with col1:
            st.session_state["items_per_page"] = st.select_slider(
                "Items per page",
                options=[5, 10, 20, 50, 100],
                value=st.session_state["items_per_page"]
            )
        with col2:
            conversations_btn = st.button("Check conversations")

        # Display pagination navigation
        page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
        with page_col1:
            if st.button("← Previous") and st.session_state["page_number"] > 0:
                st.session_state["page_number"] -= 1
                st.rerun()

        with page_col2:
            st.write(f"Page {st.session_state['page_number'] + 1}")

        with page_col3:
            if st.button("Next →"):
                st.session_state["page_number"] += 1
                st.rerun()

        # Calculate offset based on pagination
        curr_offset = st.session_state["page_number"] * st.session_state["items_per_page"]
        curr_limit = st.session_state["items_per_page"]

        if filter_ids:
            filter_ids = [id.strip() for id in filter_ids.split('\n')]

        if conversations_btn:
            list_conversations(ids = filter_ids, offset = curr_offset, limit = curr_limit)

    with t_conv_msgs:
        st.markdown("**Conversation messages**")

        conversation_id = st.text_input(
            label = "Conversation ID",
            placeholder = "Ex. conversation_id"
        )

        include_metadata = st.checkbox(
            label="Include metadata",
            value=False,
        )

        if st.button("Get messages", type="primary"):
            if not conversation_id:
                st.warning("Please enter a conversation ID")
            else:
                fetch_messages(conversation_id, include_metadata)

    with t_exp_conversations:
        st.markdown("**Export Conversations Metadata to CSV**")

        filename = st.text_input(
            label = "export filename",
            placeholder = "Ex. conversations",
            max_chars = 20,
            help = "Name of the exported file without extension"
        )

        filter_ids = st.text_area(
            label = "conversation ids",
            placeholder = "conversation_id1\nconversation_id2\n... (optional)",
            help = "Conversation ids on each line"
        )

        if st.button("Export conversations"):
            if not filename:
                st.warning("Please enter a filename")
            else:
                export_conversations(filter_ids, filename)

    with t_exp_msgs:
        st.markdown("**Export Conversation Messages to CSV**")

        filename = st.text_input(
            label = "export filename",
            placeholder = "Ex. messages",
            max_chars = 20,
            help = "Name of the exported file without extension"
        )

        if st.button("Export messages"):
            if not filename:
                st.warning("Please enter a filename")
            else:
                export_messages(filename)
