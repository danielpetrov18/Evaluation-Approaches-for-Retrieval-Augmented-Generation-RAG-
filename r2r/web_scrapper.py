import re
import requests
from langchain_community.document_loaders import WebBaseLoader

class WebScrapper:

    def fetch_documents(self, urls: list[str]):
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
             
        web_loader = WebBaseLoader(        # https://www.comet.com/site/blog/langchain-document-loaders-for-web-data/
            web_paths=urls,            # Where to scrape from
            requests_per_second=5, 
            default_parser="lxml",     # A more sophisticated parser (more flexible and efficient than html.parser)
            raise_for_status=True,     # If something goes wrong an internal exception is raised
            continue_on_failure=True,  # If a page cannot be scrapped it will be skipped
        )
        docs = web_loader.load()
       
        for doc in docs:
            if doc.page_content is not None:
                doc.page_content = self.__remove_empty_lines(doc.page_content)
        
        return docs

    def __remove_empty_lines(self, text):
        """
        Remove empty lines from given text.

        This function takes a string as input and removes empty lines from it.
        It does this by using a regular expression to find and remove empty lines.
        The regular expression ``r'^\n+|\n+(?=\n)'`` will match any line that is
        either only a newline character (``\n``) or a sequence of newline
        characters that is followed by another newline character.

        Args:
            text (str): The text to remove empty lines from.

        Returns:
            str: The text with empty lines removed.
        """
        return re.sub(r'^\n+|\n+(?=\n)', '', text)        
        
    def __url_exists(self, url):
        """
        Private method to check if a URL exists or not.

        Args:
            url (str): URL to check.

        Returns:
            bool: True if URL exists, False otherwise.
        """
        try:
            response = requests.head(url, allow_redirects=True, timeout=3)
            return response.status_code < 400
        except requests.RequestException:
            return False