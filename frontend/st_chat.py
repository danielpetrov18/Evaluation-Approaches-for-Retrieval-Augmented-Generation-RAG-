import sys
import asyncio
import streamlit as st
from pathlib import Path
from app import load_client
from r2r import R2RException

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from llm import LLMHandler
#from utility.stream import StreamHandler  

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_async(coro):
    return loop.run_until_complete(coro)

# def sync_generator(async_gen: AsyncGenerator):
#     """Convert an async generator to a sync generator."""
#     async def get_next():
#         try:
#             return await anext(async_gen)
#         except StopAsyncIteration:
#             return None

#     while True:
#         item = run_async(get_next())
#         if item is None:
#             break
#         yield item

st.title("💬 Chatbot")

client = load_client()
llm = LLMHandler(client=client)
#stream = StreamHandler()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "How can I help you today?"
        }
    ]

if "rag_parameters" not in st.session_state:
    st.session_state["rag_parameters"] = {
        "temperature": 0.1,
        "top_p": 1.0,
        "max_length": 1024
    }
    
if "form_index" not in st.session_state:
    st.session_state["form_index"] = 0

with st.sidebar:
    st.subheader('Customize LLM parameters')

    st.session_state['form_index'] += 1 # Without this index, each time the app is started I get an widget key duplicate error
    with st.form(f"llm_params_form_{st.session_state['form_index']}"):
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
            help="Nucleus sampling."
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
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "How can I help you today?"
            }
        ]
        llm.clear_chat_history()
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "😎"):
        st.write(msg["content"])

prompt = st.chat_input(placeholder="Please enter your question here ...")
if prompt:
    st.session_state.messages.append(
        {
            "role": "user", 
            "content": prompt
        }
    )
    with st.chat_message("user", avatar="😎"):
        st.write(prompt)

    rag_generation_config = {
        "temperature": st.session_state.rag_parameters["temperature"],
        "top_p": st.session_state.rag_parameters["top_p"],
        "max_tokens_to_sample": st.session_state.rag_parameters["max_length"]
    }

    with st.spinner("Generating response ..."):
        try:
            rag_rsp = run_async(
                llm.rag(
                    query=prompt,
                    temperature=rag_generation_config["temperature"],
                    top_p=rag_generation_config["top_p"],
                    max_tokens_to_sample=rag_generation_config["max_tokens_to_sample"]
                )
            )
            
            st.session_state.messages.append(
                {
                    "role": "assistant", 
                    "content": rag_rsp
                }
            )
            
            with st.chat_message("assistant", avatar="🦙"):
                st.write(rag_rsp)
        except R2RException as r2re:
            st.error(f"An error occurred: {str(r2re)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")