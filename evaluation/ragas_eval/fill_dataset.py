"""
This script will be used after synthetic test data has been created using RAGAs.
    (input, reference, reference_contexts)
R2R will be used to generate the actual response and retrieval contexts.
The generated dataset will be augmented and saved into a jsonl file.
"""

import os
import sys
import json
import tempfile
from typing import List, Final, Dict, Union

from dotenv import load_dotenv
from r2r import R2RClient, R2RException
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader

# pylint: disable=C0103
# pylint: disable=C0301

load_dotenv("../../env/rag.env")

CLIENT = R2RClient(
    base_url="http://localhost:7272",
    timeout=600
)

RAG_GENERATION_CONFIG: Final[Dict[str, Union[str, float, int]]] = {
    "model": f"ollama_chat/{os.getenv('CHAT_MODEL')}",
    "temperature": float(os.getenv("TEMPERATURE")),
    "top_p": float(os.getenv("TOP_P")),
    "max_tokens_to_sample": int(os.getenv("MAX_TOKENS")),
}

SEARCH_SETTINGS: Final[Dict[str, Union[bool, int, str]]] = {
    "use_semantic_search": True,
    "limit": int(os.getenv("TOP_K")),
    "offset": 0,
    "include_metadatas": False,
    "include_scores": True,
    "search_strategy": "vanilla",
    "chunk_settings": {
        "index_measure": "cosine_distance",
        "enabled": True,
        "ef_search": 80
    }
}

# You can modify this template as needed
TEMPLATE: Final[str] = """You are a helpful RAG assistant. Your task is to provide an answer given a question and context.
Please make sure the answer is complete and relevant to the question. Do not guess or speculate. 
Do not provide any information in the answer outside the context.

**IMPORTANT:
1. DO NOT USE ANY KNOWLEDGE YOU HAVE BEEN TRAINED ON.
2. BASE YOUR ANSWER ONLY ON THE CONTEXT GIVEN.
3. IF THE CONTEXT IS NOT ENOUGH TO ANSWER THE QUESTION, SAY THAT YOU CANNOT ANSWER BASED ON THE AVAILABLE INFORMATION.
4. DO NOT GUESS OR SPECULATE.
5. DO NOT INCLUDE CITATIONS OR REFERENCES TO SPECIFIC LINES OR PARTS OF THE CONTEXT.
6. ALWAYS KEEP YOUR ANSWER RELEVANT AND FOCUSED ON THE USER'S QUESTION.
7. DO NOT PROVIDE ANY ADDITIONAL INFORMATION EXCEPT THE ANSWER.
**

### CONTEXT:
{context}

### QUESTION:
{query}

### ANSWER:
"""

def check_health() -> None:
    """
    Making sure that we can reach the client.
    If not, the script terminates.
    """
    if CLIENT.system.health().results.message.lower() != "ok":
        print("R2R server is not running or cannot be reached.")
        sys.exit(1)

def delete_files() -> None:
    """
    Delete all files prior to ingestion since we test different `chunk_size` and `overlap` values.
    During the ingestion phase `R2R` uses a `RecursiveCharacterTextSplitter` to chunk the documents.
    So we need to make sure that the files get re-ingested each time, we run this script.
    """

    ids: List[str] = [document.id for document in CLIENT.documents.list().results]
    for document_id in ids:
        CLIENT.documents.delete(id=document_id)

    print("DELETION STEP COMPLETED...")

def ingest_files(folder_path: str = "data") -> None:
    """
    1. Load markdown files
    2. Write content to temp files
    3. ingest each
    4. Then auto-clean temp folder.
    """

    loader = DirectoryLoader(
        folder_path,
        glob="**/*.md",
        exclude="README.md"
    )
    docs: list[Document] = loader.load()

    with tempfile.TemporaryDirectory() as temp_dir:
        for doc in docs:
            doc_filepath: str = doc.metadata['source'].split("/")[-1]
            temp_file_path = os.path.join(temp_dir, doc_filepath)
            with open(temp_file_path, "w", encoding="utf-8") as file:
                file.write(doc.page_content)

        for file in os.listdir(temp_dir):
            if not file.endswith(".md"):
                continue

            filepath = os.path.join(temp_dir, file)
            try:
                resp = CLIENT.documents.create(
                    file_path=filepath,
                    run_with_orchestration=True
                ).results
                print(f"{filepath}: {resp.message}")
            except R2RException as r2re:
                print(f"Error when creating document from {filepath}:", r2re.message)
                sys.exit(1)

    print("INGESTION STEP COMPLETED...")

def load_goldens(filepath: str) -> List[Dict]:
    """Loads the synthetically generated goldens from a JSONL file."""
    _goldens: List[Dict] = []
    try:
        with open(file=f"./goldens/{filepath}.jsonl", mode="r", encoding="utf-8") as file:
            # Read the file line by line and parse each line as JSON
            for line in file:
                if line.strip():  # Skip empty lines
                    _goldens.append(json.loads(line))
    except FileNotFoundError:
        print(f"File `./goldens/{filepath}.jsonl` containing goldens not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL file: {e}")
        sys.exit(1)

    return _goldens

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
        test_id: str = sys.argv[2]
    except IndexError as ie:
        print("USAGE: python fill_dataset.py <goldens-filename> <test-id> (without extension)")
        sys.exit(1)

    print(f"""{'='*80}\nGenerating dataset in ./datasets/{test_id}_dataset.jsonl
TOP_K={int(os.getenv("TOP_K"))}
MAX_TOKENS_TO_SAMPLE={int(os.getenv("MAX_TOKENS"))}
CHUNK_SIZE={int(os.getenv("CHUNK_SIZE"))}
CHUNK_OVERLAP={int(os.getenv("CHUNK_OVERLAP"))}
CHAT_MODEL={os.getenv("CHAT_MODEL")}
TEMPERATURE={float(os.getenv("TEMPERATURE"))}
{'='*80}
""")

    check_health()

    delete_files()

    ingest_files()

    goldens: List[Dict] = load_goldens(goldens_filepath)

    for i, golden in enumerate(goldens):
        try:
            response = CLIENT.retrieval.rag(
                query=golden["user_input"],
                rag_generation_config=RAG_GENERATION_CONFIG,
                search_mode="custom",
                search_settings=SEARCH_SETTINGS,
                task_prompt=TEMPLATE
            ).results

            actual_output: str = response.completion
            retrieved_contexts: List[str] = [
                chunk.text
                for chunk in response.search_results.chunk_search_results
            ]

            # If deepseek-r1 is used regardless of parameters count
            # remove the content between the <think> tags
            if "deepseek-r1" in os.getenv("CHAT_MODEL"):
                actual_output = extract_deepseek_response(actual_output)

            golden["response"] = actual_output
            golden["retrieved_contexts"] = retrieved_contexts

            print(f"Added data to sample: {i + 1} out of {len(goldens)}")

        except R2RException as r2re:
            print(f"Something went wrong when submitting query: {i} due to {str(r2re)}")
            sys.exit(1)

    # Persist the complete dataset
    os.makedirs("./datasets", exist_ok=True)  # Create the directory if it doesn't exist
    with open(file=f"./datasets/{test_id}_dataset.jsonl", mode="w", encoding="utf-8") as f:
        for golden in goldens:
            f.write(json.dumps(golden, ensure_ascii=False) + "\n")
