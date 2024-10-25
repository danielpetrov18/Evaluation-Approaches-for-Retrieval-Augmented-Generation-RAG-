import re
import logging
import requests
import html2text
from typing import List, Optional
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader

class Scraper:
    def __init__(
        self,
        requests_per_second: int = 5,
        max_retries: int = 3,
        timeout: int = 5
    ):
        """
        Args:
            requests_per_second: Rate limiting for requests
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.__loader = WebBaseLoader(
            requests_per_second=requests_per_second,
            default_parser="lxml",
            raise_for_status=True,
            continue_on_failure=True,
        )
        self.max_retries = max_retries
        self.timeout = timeout
        self.excluded_patterns = [
            r'cookie policy',
            r'privacy notice',
            r'terms of service',
            r'navigation menu',
            r'copyright ©'
        ]
        
        # Initialize HTML to text converter
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        self.h2t.ignore_emphasis = True
        
        logging.basicConfig(level=logging.DEBUG)
        self.__logger = logging.getLogger(__name__)
        
    def fetch_documents(self, urls: List[str]) -> List[Document]:        
        """
        Fetch and process documents from the provided URLs.

        Args:
            urls (List[str]): List of URLs to fetch documents from.

        Returns:
            List[Document]: List of processed documents.

        The function validates the provided URLs, fetches the content from valid URLs, processes the content to remove unwanted patterns and formats it, and returns the processed documents. 
        Invalid URLs or errors during processing are logged.
        """
        if not urls:
            self.__logger.warning("No urls provided for web-scraping!")
            return []
        
        valid_urls = set()    # Avoiding replicas by using a set.
        processed_urls = set()
        
        # Take only the valid URLs.
        valid_urls = [url for url in urls if self.__validate_url(url)]
        
        # Fetch and process documents but only the valid ones.
        self.__loader.web_paths = list(valid_urls)
        docs = []
        
        raw_docs = self.__loader.load() # Scrape the data from the URLs. Messy format.
        for doc in raw_docs:
            if doc.page_content:
                try:
                    processed_doc = self.__process_document(doc) # Clean and structure the document
                    if processed_doc:
                        docs.append(processed_doc)
                except Exception as e:
                    self.__logger.warn(f"Error loading document: {str(doc.metadata['source'])}")

        return docs

    def __validate_url(self, url: str) -> bool:
        for attempt in range(self.max_retries):
            try:
                response = requests.head(url, 
                                      allow_redirects=True, 
                                      timeout=self.timeout
                                    )
                return response.status_code < 400 
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    self.logger.warning(f"Failed to validate URL {url} after {self.max_retries} attempts: {str(e)}")
                    return False
                continue
        return False

    def __process_document(self, doc: Document) -> Optional[Document]:
        try:
            content = self.h2t.handle(doc.page_content) # Convert HTML to markdown-like text
            content = self.__clean_content(content)     # Remove whitespaces and newlines and excluded patterns
            doc.page_content = content                  # Update the page content
            return doc
        except Exception as e:
            self.logger.error(f"Error processing document {doc.metadata['source']}: {str(e)}")
            return None

    def __clean_content(self, content: str) -> str:
        # Remove excluded patterns
        for pattern in self.excluded_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove empty lines
        content = re.sub(r'^\s*$\n', '', content, flags=re.MULTILINE)
        
        return content.strip()