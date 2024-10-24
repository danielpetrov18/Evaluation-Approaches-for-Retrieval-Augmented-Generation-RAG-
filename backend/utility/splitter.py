import logging
from os import getenv
from typing import List, Optional
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Splitter:
    def __init__(self):
        CHUNK_SIZE = int(getenv("CHUNK_SIZE", 512))
        CHUNK_OVERLAP = int(getenv("CHUNK_OVERLAP", 128))
        separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "? ",    # Questions
            "! ",    # Exclamations
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ]
                
        logging.basicConfig(level=logging.DEBUG)
        self.__logger = logging.getLogger(__name__)
        self.__recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=separators
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving context and structure.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents with preserved metadata
        """
        if not documents:
            self.__logger.warning("No documents provided for splitting!")
            return []
        
        processed_chunks = []
        for doc in documents:
            try:
                if not doc.page_content or not doc.page_content.strip():
                    self.__logger.warning(f"Empty document found: {doc.metadata.get('source', 'unknown source')}")
                    continue
                
                chunks = self.__recursive_splitter.split_documents([doc])
                processed_chunks.extend(chunks)           
            except Exception as e:
                self.logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown source')}: {str(e)}")
                continue
        return processed_chunks

    def __merge_metadata(self, original_metadata: dict, new_metadata: dict) -> dict:
        """Merge original document metadata with new chunk metadata."""
        metadata = original_metadata.copy()
        metadata.update(new_metadata)
        return metadata