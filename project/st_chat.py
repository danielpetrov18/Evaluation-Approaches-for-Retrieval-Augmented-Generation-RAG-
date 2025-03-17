"""Main page of the app where one can interact with the LLM."""

# pylint: disable=E0401

import streamlit as st
from st_app import load_client
from utility.r2r.retrieval import (
    retrieve_conversation,
    submit_query
)

if __name__ == "__page__":
    st.title("💬 Chatbot")

    with st.sidebar:
        if st.session_state['conversation_id']:
            st.markdown(
                f"**Selected conversation:**<br>{st.session_state['conversation_id']}",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "**No conversation selected**",
                unsafe_allow_html=True
            )

        selected_conversation_id = st.text_input(
            label="Pick a conversation",
            placeholder="Ex. conversation_id",
            key="conversation_id_input",
            help="By selecting an existing one, the context and previous interactions are loaded"
        )

        if st.button("Confirm", type="primary", key="conv_id_btn"):
            if not selected_conversation_id:
                st.warning("Please enter a conversation ID")
            else:
                # Conversation contains a sequence of all messages so far
                # Make sure that the conversation exists
                # And if it's the same id, don't update it
                conversation = retrieve_conversation(load_client(), selected_conversation_id)
                if conversation and st.session_state['conversation_id'] != selected_conversation_id:
                    st.session_state['conversation_id'] = selected_conversation_id
                    st.rerun()

    if not st.session_state['conversation_id']:
        st.info("Select a conversation or submit a query to get started")
    else:
        messages = retrieve_conversation(load_client(), st.session_state['conversation_id'])
        for element in messages:
            role = element.message.role.upper()
            content = element.message.content

            with st.chat_message(role, avatar="🤖" if role == "assistant" else "😎"):
                st.write(content)

    query = st.chat_input(placeholder="Please enter your question here ...")
    if query:
        with st.spinner(text="Generating response ...", show_time=True):
            submit_query(load_client(), query, st.session_state['conversation_id'])
            st.rerun()
