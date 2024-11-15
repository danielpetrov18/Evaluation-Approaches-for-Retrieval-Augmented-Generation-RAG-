import sys
import time
import streamlit as st
from app import load_client
from r2r import R2RException

st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 12px 16px;
        margin-bottom: 14px;
        font-size: 14px;
        line-height: 1.5;
    }

    .stChatMessage.user {
        background-color: #f0f2f6;
        color: #333;
    }

    .stChatMessage.assistant {
        background-color: #e6f2ff;
        color: #1a73e8;
    }

    .stSpinner > div {
        border-color: #1a73e8 #1a73e8 #1a73e8 transparent;
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 16px;
        color: #333;
    }

    .sidebar-button {
        background-color: #1a73e8;
        color: #fff;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 14px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .sidebar-button:hover {
        background-color: #0b5dcc;
    }
    </style>
""", unsafe_allow_html=True)

def prompt_llm(query: str, rag_generation_config: dict):
    backend_client = load_client()
    return backend_client.prompt_llm(query, rag_generation_config)

st.title("💬 Chat")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

if "rag_parameters" not in st.session_state:
    st.session_state["rag_parameters"] = {
        "temperature": 0.1,
        "top_p": 0.95,
        "max_length": 1024
    }
            
with st.sidebar:
    st.subheader('Choose LLM parameters')

    with st.form("llm_params_form"):
        temperature = st.slider(
            'Temperature',
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_parameters["temperature"],
            step=0.01,
            help="Controls randomness: Lower values make output more focused, higher values more creative."
        )

        top_p = st.slider(
            'Top P',
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_parameters["top_p"],
            step=0.01,
            help="Nucleus sampling: Considers the smallest set of tokens with probabilities that add up to top_p."
        )

        max_length = st.slider(
            'Max Length',
            min_value=64,
            max_value=1024,
            value=st.session_state.rag_parameters["max_length"],
            step=16,
            help="Maximum number of tokens in the generated response."
        )

        submitted = st.form_submit_button("Save Parameters")
        if submitted:
            st.session_state.rag_parameters["temperature"] = temperature
            st.session_state.rag_parameters["top_p"] = top_p
            st.session_state.rag_parameters["max_length"] = max_length
            st.success("Parameters updated successfully!")

    clear_history_button = st.button("🗑️ Clear Conversation")
    if clear_history_button:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
        client = load_client()
        client.clear_history()
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "😎"):
        st.write(msg["content"])

prompt = st.chat_input(placeholder="Please enter your question here...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt}) 
    with st.chat_message("user", avatar="😎"):
        st.write(prompt)

    rag_generation_config = {
        "temperature": st.session_state.rag_parameters["temperature"],
        "top_p": st.session_state.rag_parameters["top_p"],
        "max_tokens": st.session_state.rag_parameters["max_length"]
    }
    
    with st.spinner("Generating response ..."):
        try:
            response = prompt_llm(prompt, rag_generation_config)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant", avatar="🦙"):
                st.write(response)
        except R2RException as r2re:
            st.error(f"An error occurred: {str(r2re)}")
            time.sleep(3)
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            time.sleep(3)
            st.rerun()