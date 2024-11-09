from client import R2RBackend
from utility.scraper import Scraper
from utility.splitter import Splitter

# client = R2RBackend()
# print(f'Clients health: {client.health()}')

# scraper = Scraper()
# splitter = Splitter()

# folder_path = 'data'
# client.ingest_files(folder_path)

# urls = [
#     "https://realpython.com/build-llm-rag-chatbot-with-langchain/",
#     "https://realpython.com/python313-new-features/",
#     "https://realpython.com/pandas-dataframe/"
# ]

# documents = scraper.fetch_documents(urls)
# split_documents = splitter.split_documents(documents)
    
# for url in urls:
#         chunks = [split_doc for split_doc in split_documents if split_doc.metadata['source'] == url]
        
#         try:
#             if chunks:
#                 metadata = chunks[0].metadata
#                 chunks_text = [{"text": chunk.page_content} for chunk in chunks]
#                 resp = client.ingest_chunks(chunks_text, metadata)
#                 print(resp)
#         except Exception as e:
#             print(e)

# docs_metadata = client.documents_overview()
# for doc_metadata in docs_metadata:
#     print(doc_metadata)

# doc_id = client.documents_overview()[0]['id']
# chunks = client.document_chunks(document_id=doc_id)
# [print(chunk) for chunk in chunks]

# for doc_metadata in docs_metadata:
#     if doc_metadata['title'] == 'dummy.txt':
#         doc_id = doc_metadata['id']
#         doc_title = doc_metadata['title']
#         client.document_chunks(document_id=doc_id)[0]['text']
#         print(f'Before updating file: {client.document_chunks(document_id=doc_id)[0]['text']}, version: {doc_metadata["version"]}')

# filepaths = [f"./data/{doc_title}"]
# doc_ids = [doc_id]

# client.update_files(filepaths=filepaths, document_ids=doc_ids)

# for doc_metadata in docs_metadata:
#     if doc_metadata['title'] == 'dummy.txt':
#         client.document_chunks(document_id=doc_id)[0]['text']
#         print(f'After updating file: {client.document_chunks(document_id=doc_id)[0]['text']}, version: {doc_metadata["version"]}')

# print(f'Before deleting element: {len(client.documents_overview())}')

# client.delete([{
#     "document_id": {
#         "$eq": docs_metadata[0]['id']
#         }
#     }
# ])

# print(f'After deleting element: {len(client.documents_overview())}')

#client.clean_db()
# print(client.documents_overview())

import ollama
from message import Message 
from history import ChatHistory

#embedding_response = ollama.embed(model='mxbai-embed-large', input='Hello world')
#embedding_response2 = ollama.embed(model='mxbai-embed-large', input='Good day')

# Fetching the item on 0th index since the response is a list of lists
# m1 = Message(role='user', content='Hello world', embedding=[0.0]*1024)
# m2 = Message(role='assistant', content='Good day', embedding=[0.0]*1024)	

import time
history = ChatHistory(max_size=10)
for i in range(11):
    time.sleep(1)
    history.add_message(Message(role='user', content='Hello world', embedding=[0.0]*1024))
    
[print(msg.timestamp) for msg in history.get_all_messages()]