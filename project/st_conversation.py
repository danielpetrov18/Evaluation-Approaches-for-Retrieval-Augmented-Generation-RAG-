"""GUI to manage conversations."""

# pylint: disable=E0401
# pylint: disable=C0301

import re
import streamlit as st
from st_app import r2r_client
from backend.conversation import (
    list_conversations,
    fetch_messages,
    export_conversations,
    export_messages
)

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

        col1, col2 = st.columns(spec=[3, 1])
        with col1:
            st.session_state["items_per_page"] = st.select_slider(
                "Items per page",
                options=[5, 10, 20, 50, 100],
                value=st.session_state["items_per_page"]
            )
        with col2:
            conversations_btn = st.button("Check conversations", key="list_conv_btn")

        page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
        with page_col1:
            if st.button("← Previous", key="prev_page"):
                if st.session_state["page_number"] > 0:
                    st.session_state["page_number"] -= 1
                    st.rerun()
                else:
                    st.info("This is the first page")

        with page_col2:
            st.write(f"Page {st.session_state['page_number'] + 1}")

        with page_col3:
            if st.button("Next →"):
                st.session_state["page_number"] += 1
                st.rerun()

        offset = st.session_state["page_number"] * st.session_state["items_per_page"]
        limit = st.session_state["items_per_page"]

        if filter_ids:
            filter_ids = [id.strip() for id in filter_ids.split('\n')]

        if conversations_btn:
            list_conversations(r2r_client(), filter_ids, offset, limit)

    with t_conv_msgs:
        st.markdown("**Conversation messages**")

        conversation_id = st.text_input(
            label = "Conversation ID",
            placeholder = "Ex. conversation_id",
            value=""
        )

        include_metadata = st.checkbox(
            label="Include metadata",
            value=False
        )

        if st.button("Get messages", type="primary", key="conv_msgs_btn"):
            if not conversation_id:
                st.warning("Please enter a conversation ID")
            else:
                fetch_messages(r2r_client(), conversation_id, include_metadata)

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

        if st.button("Export conversations", key="export_conv_btn"):
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

        role_col, msg_len_col, conv_id_col = st.columns(3)

        with role_col:
            role_filter = st.selectbox(
                label="Role to filter on",
                options=["all", "user", "assistant"],
                help="Leaving it to all selects all"
            )

        with msg_len_col:
            msg_len_filter = st.select_slider(
                label="Minimum message length",
                options=[0, 50, 100, 200, 500, 1000],
                value=0,
                help="Leaving it to 0 selects all"
            )

        with conv_id_col:
            conv_id_filter = st.text_area(
                label="Conversation ids",
                value="",
                height=100,
                placeholder="conversation_id1,conversation_id2,...",
                help="Leaving it empty selects all"
            )

        if st.button("Export messages"):
            if not filename:
                st.warning("Please enter a filename")
            else:
                filters = {}

                if role_filter != "all":
                    filters['role'] = role_filter

                if msg_len_filter > 0:
                    filters['min_msg_length'] = msg_len_filter

                if conv_id_filter:
                    clean_ids = [id.strip() for id in re.split(r'[,\n\s]+', conv_id_filter) if id.strip()]
                    filters['conversation_id'] = clean_ids

                export_messages(filename, filters)
