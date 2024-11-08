import sys
from utility.scraper import Scraper
from utility.splitter import Splitter

scraper = Scraper()
splitter = Splitter()

urls = [
    #"https://de.wikipedia.org/wiki/%C3%96sterreich", 
    #"https://en.wikipedia.org/wiki/Bulgaria",
    "https://r2r-docs.sciphi.ai/documentation/python-sdk/graphrag",
    "https://docs.ragas.io/en/stable/howtos/customizations/metrics/_write_your_own_metric/#implementation"
]

documents = scraper.fetch_documents(urls)
split_documents = splitter.split_documents(documents)
    
print(documents[0].page_content)