"""Main page of the app where one can interact with the LLM."""

# pylint: disable=E0401
# pylint: disable=C0103

import streamlit as st
from st_app import load_client
from utility.r2r.retrieval import (
    Message,
    retrieve_conversation,
    check_conversation_exists,
    add_message,
    submit_query,
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
                    # Reset the messages to load from the new conversation
                    st.session_state.messages = []
                    st.rerun()

        # A button to start a new conversation
        if st.button(
            "New Conversation",
            key="new_conv_btn",
            help="Use this if there's already a selected conversation. Otherwise submit a query."
        ):
            st.session_state['conversation_id'] = None
            st.session_state.messages = []
            st.rerun()

    # Load conversation messages if we have a conversation ID and no messages loaded yet
    if st.session_state['conversation_id'] and not st.session_state.messages:
        with st.spinner("Loading conversation..."):
            messages = retrieve_conversation(load_client(), st.session_state['conversation_id'])
            if messages:
                st.session_state.messages = messages

    if not st.session_state.messages and not st.session_state['conversation_id']:
        st.info("Select a conversation or submit a query to get started")
    else:
        for msg in st.session_state.messages:
            role = msg['role']
            content = msg['content']
            with st.chat_message(role, avatar="🤖" if role == "assistant" else "😎"):
                st.write(content)

    query = st.chat_input(placeholder="Please enter your question here ...")
    if query:
        # Display newly submitted query
        with st.chat_message("user", avatar="😎"):
            st.write(query)

        st.session_state['conversation_id'] = check_conversation_exists(
            load_client(),
            st.session_state['conversation_id']
        )

        usr_msg = {
            "role": "user",
            "content": query
        }

        # Adding the message to the conversation
        new_msg = add_message(load_client(), st.session_state['conversation_id'], usr_msg)

        # Then to session state
        if not st.session_state.messages:
            st.session_state.messages = [new_msg]
        else:
            st.session_state.messages.append(new_msg)

        # Wait for the response from LLM
        with st.spinner(text="Generating response ...", show_time=True):
            llm_response = submit_query(load_client())
            st.session_state.messages.append(
                Message(
                    role = "user",
                    content = "query"
                )
            )
            with st.chat_message("assistant", avatar="🤖"):
                st.write(llm_response)
