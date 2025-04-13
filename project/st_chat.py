"""Main page of the app where one can interact with the LLM."""

# pylint: disable=E0401
# pylint: disable=C0103

import streamlit as st
from st_app import r2r_client
from backend.chat import (
    retrieve_messages,
    check_conversation_exists,
    set_new_prompt,
    add_message,
    submit_query,
    extract_completion
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
            st.markdown("**No conversation selected**")

        selected_conversation_id = st.text_input(
            label="Pick a conversation",
            placeholder="Ex. conversation_id",
            key="conversation_id_input",
            help="By selecting an existing one, the context and previous interactions are loaded",
            value=st.session_state['conversation_id']
        )

        if st.button("Confirm", type="primary", key="conv_id_btn"):
            if not selected_conversation_id.strip():
                st.warning("Please enter a conversation ID")
            elif selected_conversation_id == st.session_state['conversation_id']:
                st.warning("Please select a different conversation")
            else:
                msgs = retrieve_messages(r2r_client(), selected_conversation_id)
                if msgs:
                    st.session_state['conversation_id'] = selected_conversation_id
                    st.session_state.messages = msgs
                    st.session_state['parent_id'] = st.session_state.messages[-1].id
                    st.rerun() # To display messages

        # A button to start a new conversation
        if st.button(
            label="New Conversation",
            key="new_conv_btn",
            help="Use this if there's already a selected conversation. Otherwise submit a query."
        ):
            st.session_state['conversation_id'] = None
            st.session_state.messages = []
            st.session_state['parent_id'] = None
            st.rerun()

        # Select a different prompt from the default one
        new_prompt_name = st.text_input(
            label="Prompt Name",
            value=st.session_state['selected_prompt'],
        )

        if st.button(label="Confirm", key="new_prompt_btn"):
            if not new_prompt_name.strip():
                st.error("Please enter a prompt name.")
            elif new_prompt_name == st.session_state['selected_prompt']:
                st.error("Please enter a different prompt name.")
            else:
                if set_new_prompt(r2r_client(), new_prompt_name.strip()):
                    st.success(body=f"Selected prompt: {new_prompt_name}")
                else:
                    st.error(body=f"Prompt: {new_prompt_name} doesn't exist!")

    # Load conversation messages if we have a conversation ID and no messages loaded yet
    if st.session_state['conversation_id'] and not st.session_state.messages:
        with st.spinner("Loading conversation..."):
            messages = retrieve_messages(r2r_client(), st.session_state['conversation_id'])
            if messages:
                st.session_state.messages = messages
                st.session_state['parent_id'] = st.session_state.messages[-1].id
            else:
                st.session_state.messages = []
                st.session_state['parent_id'] = None

    if not st.session_state.messages and not st.session_state['conversation_id']:
        st.info("Select a conversation or submit a query to get started")
    else:
        for msg in st.session_state.messages:
            role = msg.role
            content = msg.content
            with st.chat_message(role, avatar="🤖" if role == "assistant" else "😎"):
                st.write(content)

    query = st.chat_input(placeholder="Please enter your question here ...")
    if query:
        with st.chat_message("user", avatar="😎"):
            st.write(query)

        # If there was no conversation id a new conversation will be created
        check_conversation_exists(r2r_client())

        add_message(r2r_client(), {"role": "user", "content": query})

        with st.chat_message("assistant", avatar="🤖"):
            response_generator = submit_query(r2r_client())
            response = st.write_stream(extract_completion(response_generator))

        add_message(r2r_client(), {"role": "assistant", "content": response})
