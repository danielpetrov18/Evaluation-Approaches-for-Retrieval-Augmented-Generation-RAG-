import sys
import streamlit as st
from pathlib import Path
from r2r import R2RException

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from client import R2RBackend

# First time is always slow, because a connection must be established.
@st.cache_resource
def get_r2r_client():
    config_filepath = backend_dir / 'config.toml'    
    client = R2RBackend(config_filepath)
    resp = client.health()
    if resp == 'ok':
        return client
    else:
        raise Exception(resp)

st.title("🦙 Llama Chatbot")

with st.sidebar:
    st.subheader('Parameters for the LLM:')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.95, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=1_024, value=1024, step=16)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# Add a clear conversation button
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.rerun()

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
prompt = st.chat_input(key="prompt", placeholder="Ask a question ...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner('Processing your query...'):
        try:
            rag_generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_length
            }
            
            # Get response considering chat history
            response = get_r2r_client().rag(
                query=prompt,
                rag_generation_config=rag_generation_config
            )
                    
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        except R2RException as r2re:
            st.error(f"An error occurred: {str(r2re)}")
            print(e)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            print(e)