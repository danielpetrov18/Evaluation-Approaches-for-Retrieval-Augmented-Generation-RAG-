from langchain_text_splitters import RecursiveCharacterTextSplitter

class Splitter:
    
    def __init__(self):
        self.__recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, 
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False
        )
    
    def split_documents(self, documents):
        """
        Split a list of documents into chunks.

        Args:
            documents (list[Document]): List of documents to split.

        Returns:
            list[Document]: List of documents with each document split into chunks.
        """
        return self.__recursive_splitter.split_documents(documents)