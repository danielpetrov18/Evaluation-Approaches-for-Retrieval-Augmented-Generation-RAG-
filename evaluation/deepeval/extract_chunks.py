"""
Simple script that I use for extracting chunks from documents.
Those chunks are going to be used as context for generating data with DeepEval.
The algorithm I follow:
    1. I go over all the ingested documents
    2. Select 3 or len(chunks) chunks at random for each document
    3. For each chunk out ouf every document I select 4 other chunks using semantic similarity
        (out of all the documents)
    4. Finally, all 5 chunks are grouped together into a context
    
Alternatively, one can use the `generate_goldens_from_docs` in **DeepEval**.
My justification for not using it is that `R2R` uses Postgress with pgvector.
`DeepEval` uses chromadb. Additionally, they use different splitters, to chunk the documents.
So to stay as close as possible to `R2R` I went with my own approach. (simplified)
"""

# pylint: disable=C0116

import os
import sys
import json
import random
from typing import List, Tuple, Dict, Final

import numpy as np
from dotenv import load_dotenv
from ollama import Client, Options
from r2r import R2RClient, R2RException, ChunkResponse

load_dotenv("../../env/rag.env")

R2R_CLIENT = R2RClient(
    base_url="http://localhost:7272",
    timeout=600
)

OLLAMA_CLIENT = Client(host="http://localhost:11434")
OLLAMA_OPTIONS = Options(
    temperature=float(os.getenv("TEMPERATURE")),
    top_p=float(os.getenv("TOP_P")),
    top_k=int(os.getenv("TOP_K")),
    num_ctx=int(os.getenv("LLM_CONTEXT_WINDOW_TOKENS")),
)

# Retrieve this many chunks from a document at random
CHUNKS_PER_DOCUMENT: Final[int] = 3

CHUNKS_PER_CONTEXT: Final[int] = 5

# For each chunk have the id as key and compute the embedding to the text
CHUNKS_WITH_EMBEDDINGS: Dict[str, Tuple[str, List[float]]] = {}

def delete_files() -> None:
    """
    Delete all files prior to ingestion since we test different `chunk_size` and `overlap` values.
    During the ingestion phase `R2R` uses a `RecursiveCharacterTextSplitter` to chunk the documents.
    So we need to make sure that the files get re-ingested each time, we run this script.
    """

    ids: List[str] = [document.id for document in R2R_CLIENT.documents.list().results]
    for document_id in ids:
        R2R_CLIENT.documents.delete(id=document_id)

def ingest_files(folder_path: str = "data") -> None:
    """
    This will use the unstructured service behind the scenes to ingest the files.
    The `RecursiveCharacterTextSplitter` will be used to chunk the documents.
    For this implementation and the dataset I use I require all `md` files and exclude `README.md`.
    NOTE: For your custom dataset you might want to change this.
    
    Args:
        folder_path: Path to the folder containing the documents
    """
    for file in os.listdir(folder_path):
        if not file.endswith(".md") or file == "README.md":
            continue

        filepath: str = os.path.join("data", file)
        try:
            response = R2R_CLIENT.documents.create(
                file_path=filepath,
                run_with_orchestration=True
            ).results
            print(f"{filepath}: {response.message}")
        except R2RException as r2re:
            print("Error when creating document:", r2re.message)

def compute_embedding(text: str) -> List[float]:
    """
    Uses `Ollama` to convert the text to an embedding preserving the semantic meaning.
    
    Args:
        text: Text to compute the embedding for
    
    Returns:
        List of floats representing the vector embedding
    """
    return OLLAMA_CLIENT.embeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        prompt=text,
        options=OLLAMA_OPTIONS
    )["embedding"]

def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute the cosine similarity between two vector embeddings.
    
    Args:
        embedding1: First vector embedding
        embedding2: Second vector embedding
    
    Returns:
        float: Cosine similarity score between the two embeddings. 
        The closer to 1, the more similar the embeddings are.
    
    Raises:
        ValueError: If the embeddings have different lengths
    """
    if len(embedding1) != len(embedding2):
        raise ValueError("Embeddings must have the same length!")

    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_n_similar_chunks(current_chunk_id: str, n: int) -> List[str]:
    """
    It receives a `chunk_id` of the current chunk and returns the `n` most similar chunks.
    It makes use of `compute_similarity` to compute the cosine similarity between chunks.
    The closer to 1, the more similar the chunks are.
    The chunks are sorted in descending order of similarity.
    Finally, we make sure that only the `n` chunks are kept.
    
    Args:
        current_chunk_id: ID of the current chunk
        n: Number of similar chunks to retrieve
    
    Returns:
        List of chunk texts
    """
    similarities: Dict[str, float] = {}

    for chunk_id, (chunk_text, embedding) in CHUNKS_WITH_EMBEDDINGS.items():
        # We don't want to consider the same chunk relevant for the context
        if chunk_id == current_chunk_id:
            continue

        similarity: float = compute_similarity(
            embedding1 = CHUNKS_WITH_EMBEDDINGS[current_chunk_id][1],
            embedding2 = embedding
        )
        similarities[chunk_text] = similarity

    # Sort by similarity in descending order
    most_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Get the top `n` chunk texts
    top_n_chunks = [chunk_text for chunk_text, _ in most_similar[:n]]

    return top_n_chunks

def extract_context_chunks() -> List[List[str]]:
    """
    This step combines multiple others to extract the contexts.
    Each context contains chunks, which are similar to each other.
    The chunks in a given context might be derived from different documents.
    
    Returns:
        List of contexts
    """
    contexts: List[List[str]] = []

    for i, document in enumerate(R2R_CLIENT.documents.list().results):
        # Fetch all chunks of a document
        doc_chunks: List[ChunkResponse] = R2R_CLIENT.chunks.list_by_document(
            document_id=document.id
        ).results

        chunk_ids: List[str] = [chunk.id for chunk in doc_chunks]

        random_chunk_ids = list(chunk_id for chunk_id in random.sample(
                chunk_ids,
                min(CHUNKS_PER_DOCUMENT, len(chunk_ids))
            )
        )

        for j, chunk_id in enumerate(random_chunk_ids):
            initial_text: str = CHUNKS_WITH_EMBEDDINGS[chunk_id][0]
            context: List[str] = [initial_text]
            similar_chunks: List[str] = retrieve_n_similar_chunks(
                chunk_id,
                CHUNKS_PER_CONTEXT - 1
            )
            context.extend(similar_chunks)
            contexts.append(context)
            print(f"Extracted context {len(contexts)} from document {i + 1} and chunk {j + 1}")

    return contexts

if __name__ == "__main__":
    try:
        json_filepath: str = sys.argv[1]
    except IndexError as ie:
        print("USAGE: python extract_chunks.py <destination-filename> (without extension)")
        sys.exit(1)

    print(f"""{'='*80}\nGenerating context in /contexts/{json_filepath}.json.
TOP_K={int(os.getenv("TOP_K"))}
MAX_TOKENS_TO_SAMPLE={int(os.getenv("MAX_TOKENS"))}
CHUNK_SIZE={int(os.getenv("CHUNK_SIZE"))}
CHUNK_OVERLAP={int(os.getenv("CHUNK_OVERLAP"))}
CHAT_MODEL={os.getenv("CHAT_MODEL")}
TEMPERATURE={float(os.getenv("TEMPERATURE"))}
{'='*80}
""")

    delete_files()
    print("DELETION STEP COMPLETED...")

    ingest_files()
    print("INGESTION STEP COMPLETED...")

    # For every chunk in the database get its text 2 embedding mapping
    for chunk_obj in R2R_CLIENT.chunks.list().results:
        CHUNKS_WITH_EMBEDDINGS[chunk_obj.id] = (
            chunk_obj.text,
            compute_embedding(chunk_obj.text)
        )

    context_chunks: List[List[str]] = extract_context_chunks()
    print("EXTRACTION STEP COMPLETED...")

    with open(f"./contexts/{json_filepath}.json", "w", encoding="utf-8") as f:
        json.dump(
            context_chunks,
            f,
            ensure_ascii=False,
            indent=4
        )
    print("SAVED TO JSON FILE...")
