"""This module is used to scrape data from the internet asynchronously."""

from html2text import HTML2Text
from readability import Document as ReadabilityDocument
from langchain_core.documents import Document
from langchain_community.document_loaders import AsyncHtmlLoader

class AsyncScraper:
    """
    Class that encapsulates the logic to scrape data from the internet asynchronously.
    """
    def __init__(self, urls: list[str]):
        self._urls = self._remove_duplicate_urls(urls)
        self._loader = AsyncHtmlLoader(
            web_path=self._urls,
            default_parser="lxml"
        )

        # Different options: https://github.com/Alir3z4/html2text/blob/master/docs/usage.md
        self._h2t = HTML2Text()
        self._h2t.body_width = 0 # Wrap long lines
        self._h2t.unicode_snob = True # Keep unicode characters to preserve non-ASCII text
        self._h2t.escape_snob = True # Escape special characters to ensure data integrity
        self._h2t.skip_internal_links = True # Ignore internal links (e.g., "#local-anchor")
        self._h2t.ignore_links = True # Ignores links
        self._h2t.ignore_images = True # Ignores images
        self._h2t.ignore_emphasis = True # Ignore emphasis (e.g., bold, italic)
        self._h2t.bypass_tables = False # Keep markdown table structure
        self._h2t.ignore_tables = True # Ignore table-related tags while keeping the row content
        self._h2t.single_line_break = True # Single line break rather than two to reduce whitespace
        self._h2t.mark_code = True # Wrap code blocks with [code] tags to preserve code formatting
        self._h2t.wrap_tables = True
        self._h2t.wrap_list_items = True
        self._h2t.decode_errors = 'replace' # Handle decoding errors by replacing invalid characters
        self._h2t.open_quote = '"'
        self._h2t.close_quote = '"'

    def _remove_duplicate_urls(self, urls: list[str]) -> list[str]:
        return list(set(urls))

    async def fetch_documents(self) -> list[Document]:
        """
        Fetch documents from the provided URLs asynchronously.

        Returns:
            list[Document]: A list of scraped and transformed documents.
        """
        documents = await self._loader.aload()
        self._transform_documents(documents)
        return documents

    def _transform_documents(self, documents: list[Document]):
        for document in documents:
            readability_doc = ReadabilityDocument(document.page_content)
            document.page_content = readability_doc.summary()
            document.page_content = self._h2t.handle(document.page_content)
            document.page_content = document.page_content.strip()
