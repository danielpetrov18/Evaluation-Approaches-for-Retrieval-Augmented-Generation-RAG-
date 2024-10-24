import sys
import streamlit as st
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from r2r_backend import R2RBackend                     

# There's no user-specific state to maintain
# Don't need  multiple connections for each user.
# The connection should be thread-safe to avoid data races or deadlocks.
# Maintain the same client across script reruns.
@st.cache_resource
def get_r2r_client():
    config_filepath = backend_dir / 'config' / 'r2r.toml'    
    return R2RBackend(config_filepath)

st.title("💬 Chatbot")

# Using session_state to store all the data for the chatbot.
# Messages and chat history should be user specific.
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display all available messages so far.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# TODO: Think about adding a history to the prompt
prompt = st.chat_input(key="prompt", placeholder="Ask a question ...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    print("user")
    with st.spinner('Processing your query...'):
        try:
            response = get_r2r_client().rag(query=prompt)
            response_content = response['completion']['choices'][0]['message']['content']
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            st.chat_message("assistant").write(response_content)
        except Exception as e:
            print(e)
            raise Exception(e)
    