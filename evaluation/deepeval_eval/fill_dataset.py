"""
This script will be used after synthetic test data has been created using DeepEval.
    (user_input, expected_output, context)
R2R will be used to generate the actual response and retrieval context.
The generated dataset will be augmented and saved into a json file.
"""

import os
import sys
import json
from typing import Final, List

from dotenv import load_dotenv
from r2r import R2RClient, R2RException

# pylint: disable=C0301

# You can customize the template however you like
TEMPLATE: Final[str] = """You are a helpful RAG chatbot assistant. Your task is to provide an answer given a question and context.

**IMPORTANT:
1. DO NOT USE ANY KNOWLEDGE YOU HAVE BEEN TRAINED ON.
2. BASE YOUR ANSWER ONLY ON THE CONTEXT GIVEN.
3. IF THE CONTEXT IS NOT ENOUGH TO ANSWER THE QUESTION, SAY THAT YOU CANNOT ANSWER BASED ON THE AVAILABLE INFORMATION.
4. DO NOT GUESS OR SPECULATE.
5. DO NOT INCLUDE CITATIONS OR REFERENCES TO SPECIFIC LINES OR PARTS OF THE CONTEXT.
6. ALWAYS KEEP YOUR ANSWER RELEVANT AND FOCUSED ON THE USER'S QUESTION. DO NOT PROVIDE ANY ADDITIONAL INFORMATION EXCEPT THE ANSWER.
7. DO NOT PROVIDE ANY ADDITIONAL INFORMATION EXCEPT THE ANSWER.
**

### CONTEXT:
{context}

### QUESTION:
{query}

### ANSWER:
"""

def extract_deepseek_response(full_response):
    """
    Extract the actual response from deepseek-r1 output by ignoring the <think>...</think> section.
    """
    if "</think>" not in full_response:
        raise ValueError("Response from deepseek-r1 is not full!")

    strings: List[str] = full_response.split("</think>")
    answer_without_section: str = strings[-1].lstrip()
    return answer_without_section

if __name__ == "__main__":
    try:
        goldens_filepath: str = sys.argv[1]
    except IndexError as ie:
        print("USAGE: python fill_dataset.py <goldens-filename> (without extension)")
        sys.exit(1)

    load_dotenv("../../env/rag.env")

    client = R2RClient(
        base_url="http://localhost:7272",
        timeout=600
    )

    if client.system.health().results.message.lower() != "ok":
        print("R2R server is not running or cannot be reached.")
        sys.exit(1)

    rag_generation_config = {
        "model": f"ollama_chat/{os.getenv("CHAT_MODEL")}",
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
    try:
        with open(file=f"./datasets/{goldens_filepath}.json", mode="r", encoding="utf-8") as f:
            goldens = json.load(f)
    except FileNotFoundError:
        print(f"File `./datasets/{goldens_filepath}.json` containing goldens not found.")
        sys.exit(1)

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

            # If deepseek-r1 is used regardless of parameters count
            # remove the content between the <think> tags
            if "deepseek-r1" in os.getenv("CHAT_MODEL"):
                actual_output = extract_deepseek_response(actual_output)

            golden["actual_output"] = actual_output
            golden["retrieval_context"] = retrieval_context

            print(f"Added data to sample: {i + 1} out of {len(goldens)}")

        except R2RException as r2re:
            print(f"Something went wrong when submitting query: {i} due to {str(r2re)}")

    # # Save the file into a json file
    with open(file=f"./datasets/full-{goldens_filepath}.json", mode="w", encoding="utf-8") as f:
        json.dump(goldens, f, ensure_ascii=False, indent=4)
