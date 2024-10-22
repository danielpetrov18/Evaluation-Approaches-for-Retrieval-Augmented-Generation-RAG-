import re
import requests
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader

"""
    Same as with the Splitter class I decided to create a separate class for the scraper
    just so that one can reuse it multiple times without having to instantiate a new object.
"""

class Scraper:

    def __init__(self):
        # https://www.comet.com/site/blog/langchain-document-loaders-for-web-data/
        self.__loader = WebBaseLoader( 
            requests_per_second=5, 
            default_parser="lxml",     # A more sophisticated parser (more flexible and efficient than html.parser)
            raise_for_status=True,     # If something goes wrong an internal exception is raised
            continue_on_failure=True,  # If a page cannot be scrapped it will be skipped
        )

    def fetch_documents(self, urls: list[str]) -> list[Document]:
        """
        Fetch documents from given URLs.

        This method takes a list of URLs, and uses the WebBaseLoader to fetch the documents from valid URLs. 
        It then removes empty lines from the page content of each document and returns the list of documents.

        Args:
            urls (list[str]): List of URLs to fetch documents from.

        Returns:
            list[Document]: List of documents fetched from the URLs.
        """
        valid_urls = [url for url in urls if self.__url_exists(url)] # Consider only existing URLs    
        self.__loader.web_paths = valid_urls # Provide target to scrape data from
        docs = self.__loader.load()
       
        for doc in docs:
            if doc.page_content is not None:
                doc.page_content = self.__remove_empty_lines(doc.page_content)
        
        return docs

    def __remove_empty_lines(self, text: str):
        return re.sub(r'^\n+|\n+(?=\n)', '', text)        
        
    def __url_exists(self, url: str):
        try:
            response = requests.head(url, allow_redirects=True, timeout=3)
            return response.status_code < 400
        except requests.RequestException:
            return False