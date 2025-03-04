import os
import logging
from typing import List
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Splitter:
    
    def __init__(self):
        CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
        CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
        
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
                
        self._logger = logging.getLogger(__name__)
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=separators
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller chunks using a recursive character-based text splitter.
        
        Args:
            documents (List[Document]): A list of Document objects to be split. Each Document should have a non-empty page_content.
        
        Returns:
            List[Document]: A list of Document objects representing the chunks. Each chunk retains the metadata of the original document.
        """
        if not documents:
            self._logger.warning("[-] No documents provided for splitting! [-]")
            return []
        
        processed_chunks = []
        for doc in documents:
            try:
                if not doc.page_content or not doc.page_content.strip():
                    self._logger.warning(f"[-] Empty document found: {doc.metadata.get('source', 'unknown source')}! [-]")
                    continue
                
                chunks = self._recursive_splitter.split_documents([doc])
                processed_chunks.extend(chunks)           
            except Exception as e:
                self._logger.error(f"[-] Error splitting document {doc.metadata.get('source', 'unknown source')}: {str(e)} [-]")
                continue
        return processed_chunks