from langchain_text_splitters import RecursiveCharacterTextSplitter

class IOHelper:
    
    def split_docs(self, docs):
        # Check out the config file of R2R (section: ingestion)
        """
        Split a list of documents into chunks.

        The splitting process is done according to the RecursiveCharacterTextSplitter
        algorithm from the langchain_text_splitters library. The chunk size and overlap
        are set to the default values of 512 and 20, respectively.

        Args:
            docs (list[Document]): List of documents to split.

        Returns:
            list[Document]: List of documents with each document split into chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=128,
            length_function=len,
            is_separator_regex=False
        )
        
        return splitter.split_documents(docs)
    
    def get_url_chunks(self, url, pieces):
        """
        Get chunks of a given URL from a list of document pieces.

        Args:
            url (str): URL to get chunks of.
            pieces (list[Document]): List of document pieces to search in.

        Returns:
            list[str]: List of chunks of the given URL.
        """
        url_chunks = [piece.page_content for piece in pieces if piece.metadata["source"] == url] 
        return url_chunks
    
    def get_url_metadata(self, url, pieces):
        """
        Get metadata of a given URL from a list of document pieces.

        Args:
            url (str): URL to get metadata of.
            pieces (list[Document]): List of document pieces to search in.

        Returns:
            dict[str]: Metadata of the given URL, or None if not found.
        """
        for piece in pieces:
            if piece.metadata["source"] == url:
                return piece.metadata
        return None