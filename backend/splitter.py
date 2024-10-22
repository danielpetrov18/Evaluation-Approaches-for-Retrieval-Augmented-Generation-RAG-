from os import getenv
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
    The reason why I create a separate class for the recursive splitter is so it can 
    be reused multiple times without the need to instantiate a new object. 
"""

class Splitter:
    
    def __init__(self):
        CHUNK_SIZE = int(getenv("CHUNK_SIZE", 512))
        CHUNK_OVERLAP = int(getenv("CHUNK_OVERLAP", 20))
        
        self.__recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
    
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of documents into chunks. 

        Args:
            documents (list[Document]): List of documents to split.

        Returns:
            list[Document]: List of documents with each document split into chunks.
        """
        return self.__recursive_splitter.split_documents(documents)