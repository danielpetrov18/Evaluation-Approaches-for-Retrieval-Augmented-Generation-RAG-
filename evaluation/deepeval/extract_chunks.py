"""
Simple script that I use for extracting chunks from documents.
Those chunks are going to be as context for generating data with DeepEval.
"""

import os
import sys
import json
from typing import List
from r2r import R2RClient, R2RException

CLIENT = R2RClient(
    base_url="http://localhost:7272",
    timeout=600
)

def ingest_files(folder_path: str = "data") -> None:
    """Ingest all files from the folder."""

    for file in os.listdir(folder_path):
        if file.endswith(".md") and file != "README.md":
            filepath: str = os.path.join("data", file)

            try:
                response = CLIENT.documents.create(
                    file_path=filepath,
                    run_with_orchestration=True
                ).results
                print(f"{filepath}: {response.message}")
            except R2RException as r2re:
                print("Error when creating document:", r2re.message)

def extract_context_chunks() -> List[List[str]]:
    """Extracts all chunks from the documents and groups them together."""

    document_chunks: List[List[str]] = []

    for document in CLIENT.documents.list().results:
        chunks = CLIENT.chunks.list_by_document(
            document_id=document.id
        ).results
        chunks_txt = [chunk.text for chunk in chunks]
        document_chunks.append(chunks_txt)

    return document_chunks

if __name__ == "__main__":
    try:
        json_filepath: str = sys.argv[1]
    except IndexError as ie:
        print("USAGE: python extract_chunks.py <path> (without extension)")
        sys.exit(1)

    ingest_files()
    print("INGESTION STEP COMPLETED...")

    context_chunks: List[List[str]] = extract_context_chunks()
    print("EXTRACTION STEP COMPLETED...")

    with open(f"{json_filepath}.json", "w", encoding="utf-8") as f:
        json.dump(
            context_chunks,
            f,
            ensure_ascii=False,
            indent=4
        )
    print("SAVED TO JSON FILE...")
