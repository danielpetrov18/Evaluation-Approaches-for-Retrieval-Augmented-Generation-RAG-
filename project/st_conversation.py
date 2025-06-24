# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=E0401

import streamlit as st

from backend.conversation import list_conversations, fetch_messages

if __name__ == "__page__":
    st.title("🗪 Manage conversations")

    with st.sidebar:
        st.markdown("""
### About Conversations

A `conversation` in R2R tracks the full interaction history between a user and the system, including messages from both sides.

**Use this page to:**
- View a list of recent conversations.
- Inspect the full message history for each one.
- Delete outdated or test conversations.

---

**Tip**: You can fetch up to 1000 recent conversations using the “Check conversations” button.

To explore the message history for a specific conversation, paste its ID into the second tab.
""")

    t_list, t_conv_msgs = st.tabs(["List Conversations", "Conversation messages"])

    with t_list:
        if st.button("Check conversations", key="list_conv_btn"):
            list_conversations()

    with t_conv_msgs:
        conversation_id: str = st.text_input(
            label = "Please enter conversation ID",
            placeholder = "Ex. conversation_id",
            value=""
        )

        if st.button("Get messages", type="primary", key="conv_msgs_btn"):
            if not conversation_id:
                st.warning("Please enter a conversation ID")
            else:
                fetch_messages(conversation_id.strip())
