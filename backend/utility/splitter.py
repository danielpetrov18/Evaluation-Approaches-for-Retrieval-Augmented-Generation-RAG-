"""
This module is mainly about splitting documents into smaller chunks for easier ingestion.
"""

import os
import logging
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Splitter:
    """Exposes a single method for splitting documents into smaller chunks."""

    def __init__(self):
        # pylint: disable=invalid-name
        CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
        CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

        self._separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "? ",    # Questions
            "! ",    # Exclamations
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ]
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=self._separators
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Splits a list of documents into smaller chunks.
        
        Args:
            documents (List[Document]): A list of Document objects to be split.
                Each Document should have a non-empty page_content.
        
        Returns:
            List[Document]: A list of Document objects representing the chunks. 
                Each chunk retains the metadata of the original document.
        """
        if not documents:
            self._logger.warning("[-] No documents provided for splitting! [-]")
            return []

        processed_chunks = []
        for doc in documents:
            try:
                if not doc.page_content or not doc.page_content.strip():
                    self._logger.warning(
                        "[-] Empty document found: %s! [-]", 
                        {doc.metadata.get('source', 'unknown source')}
                    )
                    continue

                chunks = self._recursive_splitter.split_documents([doc])
                processed_chunks.extend(chunks)
            except ValueError as ve:
                self._logger.warning(
                    "[-] Error splitting document %s: %s [-]", 
                    {doc.metadata.get('source', 'unknown source')}, str(ve)
                )
                continue
        return processed_chunks
