import sys
import streamlit as st
from pathlib import Path
from r2r import R2RException
from app import connect_to_backend

st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;    
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
    }
    .stChatMessage.assistant {
        background-color: #e6f2ff;
    }
    .stSpinner > div {
        border-color: #4CAF50 #4CAF50 #4CAF50 transparent;
    }
    </style>
""", unsafe_allow_html=True)

def prompt_llm(query: str, messages: list[dict], rag_generation_config: dict):
    backend_client = connect_to_backend()
    stream = backend_client.prompt_llm(query, messages, rag_generation_config)
    return stream

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir)) 
from stream_handler import R2RStreamHandler
stream_handler = R2RStreamHandler()

st.title("💬 Chat")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

if "rag_parameters" not in st.session_state:
    st.session_state.rag_parameters = {
        "temperature": 0.1,
        "top_p": 0.95,
        "max_length": 1024
    }

with st.sidebar:    
    with st.form(key="rag_params"):
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
        st.rerun()

# Display all message up to now
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🦙" if msg["role"] == "assistant" else "👤"):
        st.write(msg["content"])

prompt = st.chat_input(placeholder="Please enter your question here...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt}) 
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    rag_generation_config = {
        "temperature": st.session_state.rag_parameters["temperature"],
        "top_p": st.session_state.rag_parameters["top_p"],
        "max_tokens": st.session_state.rag_parameters["max_length"]
    }
    
    try:
        with st.chat_message("assistant", avatar="🦙"):
            # Pass all previous messages as history without the last one / current one.
            stream = prompt_llm(prompt, st.session_state.messages[:-1], rag_generation_config)   
            response = st.write_stream(stream_handler.process_stream(stream))
    
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    except R2RException as r2re:
        st.error(f"An error occurred: {str(r2re)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")