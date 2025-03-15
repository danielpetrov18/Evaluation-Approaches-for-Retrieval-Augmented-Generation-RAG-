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

# pylint: disable=E0401
from conversations import Conversations  

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

def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    try:
        load_client().conversations.delete(conversation_id)
        st.session_state["page_number"] = 0
        st.success(body=f"Deleted conversation: {conversation_id}", icon="🗑️")
    except R2RException as r2re:
        st.error(f"Error deleting conversation: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def update_conversation(conversation_id: str, input_name_key: str):
    """
    Change the name of an existing conversation.
    
    Since I had problems with this particular functionality I decided to capture
    the new name from the widget since it's automatically stored in the session
    state by Streamlit.
    """
    try:
        new_name = st.session_state[input_name_key]
        update_resp = load_client().conversations.update(
            id=conversation_id,
            name=new_name
        ).results
        st.session_state["page_number"] = 0
        st.success(
            body=f"Updated name of {conversation_id} to {update_resp.name}!",
            icon="✅"
        )
    except R2RException as r2re:
        st.error(f"Error updating conversation: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

if __name__ == "__page__":
    st.title("🗪 Manage conversations")

    if "items_per_page" not in st.session_state:
        st.session_state["items_per_page"] = 10

    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 0

    t_list, t_exp_conversations, t_exp_msgs  = st.tabs(
        [
            "List Conversations",
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

    # with t_exp_conversations:
    #     st.markdown("**Export Conversations Metadata to CSV**")

    #     filename = st.text_input(
    #         label = "export filename",
    #         placeholder = "Ex. conversations",
    #         max_chars = 20,
    #         help = "Name of the exported file without extension"
    #     )

    #     filter_ids = st.text_input(
    #         label = "conversation ids",
    #         placeholder = "Ex. conversation_id1,conversation_id2,conversation_id3 (optional)",
    #         help = "Comma separated list of conversation ids"
    #     )

    #     export_conversations_btn = st.button("Export conversations")
    #     if export_conversations_btn:
    #         try:
    #             filters = {}

    #             if filter_ids:
    #                 filters['id'] = [filter_id.strip() for filter_id in filter_ids.split(',')]

    #             get_conversations_handler().export_conversations_to_csv(
    #                 bearer_token = st.session_state["token"],
    #                 out = filename,
    #                 filters = filters
    #             )
    #             st.success("Conversations exported successfully")
    #         except R2RException as r2re:
    #             st.error(f"Error exporting conversations: {str(r2re)}")
    #         except Error as e:
    #             st.error(f"Unexpected error: {str(e)}")

    # with t_exp_msgs:
    #     st.markdown("**Export Conversation Messages to CSV**")

    #     filename = st.text_input(
    #         label = "export filename",
    #         placeholder = "Ex. messages",
    #         max_chars = 20,
    #         help = "Name of the exported file without extension"
    #     )

    #     export_msgs_btn = st.button("Export messages")
    #     if export_msgs_btn:
    #         try:
    #             get_conversations_handler().export_messages_to_csv(
    #                 bearer_token = st.session_state["token"],
    #                 out = filename
    #             )
    #             st.success("Messages exported successfully")
    #         except R2RException as r2re:
    #             st.error(f"Error exporting messages: {str(r2re)}")
    #         except Error as e:
    #             st.error(f"Unexpected error: {str(e)}")