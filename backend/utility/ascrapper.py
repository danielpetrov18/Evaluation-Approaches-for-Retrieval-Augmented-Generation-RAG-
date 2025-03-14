"""This module is used to scrape data from the internet asynchronously."""

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
            document.page_content = document.page_content.strip()
