import json
from requests import get
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class WebScrapper:

    def fetch_content(self, url):
        """
        Given a URL, the function fetches the content of the webpage and returns it.
        If the status code of the response is not 200, an exception is raised.
       
        Args:
            url (str): The URL of the webpage the content of which should be fetched.

        Returns
            str: The content of the webpage.
        """
        response = get(url)
        response.raise_for_status() # If the status code is not 200, an exception is raised.
        return response.content
        
    def parse_content(self, url, response_content):
        """
        Given the content of a webpage, the function parses the content.

        Args:
            url (str): The URL of the webpage the content of which should be parsed.
            response_content (str): The content of the webpage.

        Returns
            dict: A dictionary containing the title , the main content (paragraphs and headings) and a list of links.
        """
        soup = BeautifulSoup(response_content, 'html.parser')
        
        main_section = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.find('body')
        
        content = []
        links = []
        for tag in main_section.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a']):
            if tag.name == 'a':
                href = tag.get('href')
                if href:
                    # Make sure the href is an absolute URL
                    full_href = urljoin(url, href)
                    links.append(full_href)
            else:
                text = tag.text.strip()
                if text:
                    if tag.name.startswith('h'):
                        content.append(f"\n{tag.name.upper()}: {text}\n")
                    else:
                        content.append(text)
        
        full_content = "\n".join(content)
        
        parsed_content = {
            "url": url,
            "title": soup.title.string if soup.title else f"{url}", # If no title found use the url
            "text": full_content,
            "links": links
        }
        
        return parsed_content