# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=E0401

from typing import List, Union, Dict

import streamlit as st

from backend.chat import (
    retrieve_messages,
    check_conversation_exists,
    create_conversation,
    set_new_prompt,
    add_message,
    submit_query
)

if __name__ == "__page__":
    st.title("💬 Chatbot")

    with st.sidebar:
        if st.session_state['conversation_id']:
            st.markdown(f"**Selected conversation:**  \n{st.session_state['conversation_id']}")
        else:
            st.markdown("**No conversation selected**")

        # Select a new conversation id
        selected_conversation_id: str = st.text_input(
            label="Pick a conversation",
            placeholder=st.session_state['conversation_id'],
            key="conversation_id_input",
            help="By selecting an existing one, the previous interactions are loaded",
            value=st.session_state['conversation_id']
        )

        if st.button(label="Confirm", type="primary", key="conv_id_btn"):
            if not selected_conversation_id:
                st.warning("Please enter a conversation ID")
            elif selected_conversation_id == st.session_state['conversation_id']:
                st.warning("Please select a different conversation")
            else:
                msgs: Union[List[Dict], None] = retrieve_messages(selected_conversation_id.strip())
                # If we get None, the conversation doesn't exist
                if msgs:
                    st.session_state['conversation_id'] = selected_conversation_id
                    st.session_state['messages'] = msgs
                    st.session_state['parent_id'] = st.session_state.messages[-1]['id']
                    st.rerun() # To display messages

        # A button to start a new conversation
        if st.button(
            label="New Conversation",
            key="new_conv_btn",
            help="Use this if there's already a selected conversation. Otherwise submit a query."
        ):
            st.session_state['conversation_id'] = None
            st.session_state['messages'] = []
            st.session_state['parent_id'] = None
            st.rerun()

        # Select a different prompt from the default one
        new_prompt_name: str = st.text_input(
            label="Prompt Name",
            value=st.session_state['selected_prompt'],
            help="Select your desired RAG prompt. `rag` is the default one.",
        )

        if st.button(label="Confirm", key="new_prompt_btn"):
            new_prompt_name: str = new_prompt_name.strip()
            if not new_prompt_name:
                st.error("Please enter a prompt name.")
            elif new_prompt_name == st.session_state['selected_prompt']:
                st.error("Please enter a different prompt name.")
            else:
                if set_new_prompt(new_prompt_name):
                    st.success(body=f"Selected prompt: {new_prompt_name}")
                else:
                    st.error(body=f"Prompt: {new_prompt_name} doesn't exist!")

        st.markdown("""
### About the Chatbot

This chatbot interface allows you to interact with your RAG (Retrieval-Augmented Generation) pipeline using conversations tracked by the R2R system.

Each conversation maintains a full history of user and assistant messages. Based on your selected prompt, query context is retrieved from indexed documents to enhance the LLM response.

---

**What you can do here:**

- **Pick an existing conversation** to continue where you left off.
- **Start a new conversation** if you're initiating a different topic.
- **Change the prompt** to switch between RAG templates (e.g., `rag`, `summarizer`, etc.).
- **Submit questions** using the chat input. The system will retrieve context and generate an informed response using the current conversation state.

---

**Pipeline Overview:**

1. **Query**: You send a message to the assistant.
2. **Context Retrieval**: A semantic search + reranker selects relevant chunks from your documents.
3. **Augmentation**: The query is combined with the top context using a prompt template.
4. **LLM Generation**: The assistant produces an answer based on the enriched input.

---

- **Tip**: Use short, focused queries for best results. If context is irrelevant or missing, you can update your documents in the storage tab.

- **Note**: This is a naive RAG implementation. So no complex context filtering takes place. Noise can occur and irrelevant information can be included.
""")

    # Load conversation messages if we have a conversation ID and no messages loaded yet
    if st.session_state['conversation_id'] and not st.session_state['messages']:
        with st.spinner("Loading conversation..."):
            messages: Union[List[Dict[str, str]], None] = retrieve_messages(
                st.session_state['conversation_id']
            )
            if messages:
                st.session_state['messages'] = messages
                st.session_state['parent_id'] = st.session_state.messages[-1]['id']
            else:
                st.session_state.messages = []
                st.session_state['parent_id'] = None

    if not st.session_state.messages and not st.session_state['conversation_id']:
        st.info("Select a conversation or submit a query to get started")
    else:
        for msg in st.session_state.messages:
            role: str = msg['role']
            content: str = msg['content']
            with st.chat_message(role, avatar="🤖" if role == "assistant" else "😎"):
                st.write(content)

    query: Union[str, None] = st.chat_input(placeholder="Please enter your question here ...")
    if query:
        with st.chat_message("user", avatar="😎"):
            st.write(query)

        if not check_conversation_exists():
            create_conversation()

        add_message({"role": "user", "content": query})

        with st.chat_message("assistant", avatar="🤖"):
            response: str = submit_query()
            st.write(response)

        add_message({"role": "assistant", "content": response})
