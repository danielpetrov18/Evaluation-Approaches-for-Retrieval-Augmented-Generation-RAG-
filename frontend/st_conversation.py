"""Showcases settings and system information."""

import sys
from pathlib import Path
from r2r import R2RException
import streamlit as st
from streamlit.errors import Error
from st_app import load_client

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from conversations import Conversations

@st.cache_resource
def get_conversations_handler():
    """Get the conversations handler."""
    return Conversations(client=load_client())

if __name__ == "__page__":
    st.title("🗪 Manage conversations")

    if "token" not in st.session_state:
        bearer_token = load_client().users.login(
            email = "admin@example.com",
            password = "change_me_immediately"
        ).results.access_token.token
        st.session_state["token"] = bearer_token

    if "items_per_page" not in st.session_state:
        st.session_state["items_per_page"] = 10

    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 0

    t_list, t_exp_conversations, t_get_converstaion, t_delete_conversation, t_exp_msgs  = st.tabs(
        [
            "List Conversations",
            "Export Conversations",
            "Get Conversation",
            "Delete Conversation",
            "Export Messages"
        ]
    )

    with t_list:
        st.markdown("**List Conversations**")

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
        offset = st.session_state["page_number"] * st.session_state["items_per_page"]
        limit = st.session_state["items_per_page"]

        if conversations_btn:
            try:
                # Use pagination parameters in the API call
                conversations_response = get_conversations_handler().list_conversations(
                    offset=offset,
                    limit=limit
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

                    # Show a message if we've reached the end
                    if len(conversations) < limit:
                        st.info("You've reached the end of the conversations list.")
                else:
                    if st.session_state["page_number"] > 0:
                        st.warning("No more conversations found. Going back to previous page.")
                        st.session_state["page_number"] -= 1
                        st.rerun()
                    else:
                        st.info("No conversations found")
            except R2RException as r2re:
                st.error(f"Error checking conversations: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

    with t_exp_conversations:
        st.markdown("**Export Conversations Metadata to CSV**")

        filename = st.text_input(
            label = "export filename",
            placeholder = "Ex. conversations",
            max_chars = 20,
            help = "Name of the exported file without extension"
        )

        filter_ids = st.text_input(
            label = "conversation ids",
            placeholder = "Ex. conversation_id1,conversation_id2,conversation_id3 (optional)",
            help = "Comma separated list of conversation ids"
        )

        export_conversations_btn = st.button("Export conversations")
        if export_conversations_btn:
            try:
                filters = {}

                if filter_ids:
                    filters['id'] = [filter_id.strip() for filter_id in filter_ids.split(',')]

                get_conversations_handler().export_conversations_to_csv(
                    bearer_token = st.session_state["token"],
                    out = filename,
                    filters = filters
                )
                st.success("Conversations exported successfully")
            except R2RException as r2re:
                st.error(f"Error exporting conversations: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

    with t_get_converstaion:
        st.markdown("**Get Conversation**")

        conversation_id = st.text_input(
            label = "conversation id",
            placeholder = "Ex. conversation_id",
            max_chars = 50,
            help = "ID of the conversation"
        )

        get_conversation_btn = st.button("Get conversation")
        if get_conversation_btn and conversation_id.strip():
            try:
                conversation = get_conversations_handler().get_conversation(conversation_id).results
                for message in conversation:
                    st.json(message.model_dump())
            except R2RException as r2re:
                st.error(f"Error getting conversation: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

    with t_delete_conversation:
        st.markdown("**Delete Conversation**")

        del_conversation_id = st.text_input(
            label = "conversation id",
            placeholder = "Ex. conversation_id",
            max_chars = 50,
            help = "ID of the conversation to delete"
        )

        delete_conversation_btn = st.button("Delete conversation")
        if delete_conversation_btn and del_conversation_id.strip():
            try:
                response = get_conversations_handler().delete_conversation(del_conversation_id)
                st.success(f"Conversation deleted successfully: {response.results}")
            except R2RException as r2re:
                st.error(f"Error deleting conversation: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

    with t_exp_msgs:
        st.markdown("**Export Conversation Messages to CSV**")

        filename = st.text_input(
            label = "export filename",
            placeholder = "Ex. messages",
            max_chars = 20,
            help = "Name of the exported file without extension"
        )

        export_msgs_btn = st.button("Export messages")
        if export_msgs_btn:
            try:
                get_conversations_handler().export_messages_to_csv(
                    bearer_token = st.session_state["token"],
                    out = filename
                )
                st.success("Messages exported successfully")
            except R2RException as r2re:
                st.error(f"Error exporting messages: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")
