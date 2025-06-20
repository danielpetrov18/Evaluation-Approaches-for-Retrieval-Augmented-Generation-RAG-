"""
Prompt engineering can significantly improve the quality of responses.
It's worth trying out different prompts for RAG to see which one works best.
Prompts can be defined in YAML format. Check out the prompts page in the browser.
"""

# pylint: disable=C0301
# pylint: disable=E0401

from typing import Union

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from backend.prompt import list_prompts, create_prompt

if __name__ == "__page__":
    st.title("📝 Prompt Management")

    with st.sidebar:
        st.markdown("""
### About Prompts in R2R

A `prompt` represents a **templated instruction or query pattern** that can be reused across the system for consistent and dynamic interactions.

**Learn more**:
- [R2R Prompt API Docs](https://r2r-docs.sciphi.ai/api-and-sdks/prompts/prompts)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

### Prompt Structure in R2R

Each prompt in R2R consists of **three parts**:

- **`name`**: A unique identifier for the prompt.
- **`template`**: A string defining the structure of the prompt, with placeholders for dynamic content (e.g., `{query}`, `{context}`).
- **`input_types`**: A dictionary specifying the expected input types for each placeholder.

---

### Default Prompts

When you start with R2R, you'll have access to **16 built-in default prompts** designed for various tasks (question answering, summarization, rewriting, etc.). They are internally used by the framework.

For this project, the most relevant is:

- **`rag`** → the default Retrieval-Augmented Generation prompt, optimized for injecting retrieved context into LLM responses.

---

**You are encouraged to create your own prompts** to match the specific retrieval and generation behavior you want.
""")


    tab_list, tab_create = st.tabs(["List Prompts", "Create Prompt"])

    with tab_list:
        if st.button(label="Fetch Prompt List", key="fetch_prompts_btn"):
            list_prompts()

    with tab_create:
        with st.expander("Upload Instructions", expanded=False):
            st.markdown("""Upload a YAML file that defines a prompt.
Below is an example of a custom prompt.
Make sure you give a `unique name` to the prompt (`custom_rag` in the example below).
The `template` and `input_types` keys should be defined as shown.
""")

            st.code("""
                custom_rag:
                    template: |
                        You are a helpful assistant. Use only the information in the context below to answer the user's question.

                        Do not use any other knowledge you may have been trained on.

                        If the context does not have the information needed to answer the question, say that you cannot answer based on the available information.

                        Do not include citations or references to specific lines or parts of the context.

                        Always keep your answer relevant and focused on the user's question.

                        ### Context:
                        {context}

                        ### Query:
                        {query}

                        ## Response:
                    input_types:
                        query: str
                        context: str
            """, language="yaml", line_numbers=True, wrap_lines=True)

        uploaded_file: Union[UploadedFile, None] = st.file_uploader(
            label="Upload YAML Prompt File",
            type=["yaml", "yml"]
        )
        if st.button(label="Create Prompt", key="create_prompt_btn"):
            if uploaded_file is None:
                st.error(body="Please upload a YAML file.")
            else:
                create_prompt(uploaded_file)
