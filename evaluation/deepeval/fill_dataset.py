"""
This script will be used after synthetic test data has been created using DeepEval.
R2R will be used to generate the actual response and retrieval context.
The generated dataset will be augmented and saved into a json file.
Finally, in the Notebook `generate` the final dataset will be uploaded the to Confident AI platform.
"""

import os
import sys
import json
from dotenv import load_dotenv
from r2r import R2RClient, R2RException

# pylint: disable=C0301

# Additionally, you can use another prompt.
# If the selected prompt is not available, this prompt will be used.
TEMPLATE = """You are a helpful assistant. Use only the information in the context below to answer the user's question.

Do not use any other knowledge you may have been trained on.

If the context does not have the information needed to answer the question, say that you cannot answer based on the available information.

Do not include citations or references to specific lines or parts of the context.

Always keep your answer relevant and focused on the user's question.

### Context:
{context}

### Query:
{query}

## Response:
"""

if __name__ == "__main__":
    load_dotenv("../../env/rag.env")

    client = R2RClient(
        base_url="http://localhost:7272",
        timeout=600
    )

    if len(sys.argv) > 1:
        prompt_name = sys.argv[1]
        try:
            TEMPLATE = client.prompts.retrieve(prompt_name).results.template
            print(f"Using template: {prompt_name}")
        except R2RException:
            print(f"Template '{prompt_name}' doesn't exist. Using default template.")
    else:
        print("No template name provided. Using default template.")

    rag_generation_config = {
        "model": f"ollama/{os.getenv("CHAT_MODEL")}",
        "temperature": float(os.getenv("TEMPERATURE")),
        "top_p": float(os.getenv("TOP_P")),
        "max_tokens_to_sample": int(os.getenv("MAX_TOKENS")),
    }

    search_settings = {
        "use_semantic_search": True,
        "limit": int(os.getenv("TOP_K")),
        "offset": 0,
        "include_metadatas": False,
        "include_scores": True,
        "search_strategy": "vanilla",
    }

    goldens: list[dict] = []
    with open(file="./deepeval_dataset.json", mode="r", encoding="utf-8") as f:
        goldens = json.load(f)

    for i, golden in enumerate(goldens):
        try:
            response = client.retrieval.rag(
                query=golden["input"],
                rag_generation_config=rag_generation_config,
                search_mode="custom",
                search_settings=search_settings,
                task_prompt=TEMPLATE
            ).results

            actual_output = response.completion
            retrieval_context = [chunk.text for chunk in response.search_results.chunk_search_results]

            golden["actual_output"] = actual_output
            golden["retrieval_context"] = retrieval_context

            print(f"Added data to sample: {i + 1} out of {len(goldens)}")

        except R2RException as r2re:
            print(f"Something went wrong when submitting query: {i} due to {str(r2re)}")

    # Save the file into a json file
    with open(file="./deepeval_dataset.json", mode="w", encoding="utf-8") as f:
        json.dump(goldens, f, ensure_ascii=False, indent=4)
