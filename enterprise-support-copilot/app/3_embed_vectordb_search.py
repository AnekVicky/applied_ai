import json ,os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

#Step 1 : load the data
FILE_PATH = './enterprise-support-copilot/data/support_tickets.json'
load_dotenv()
anek_applied_ai_key = os.getenv('anek_applied_ai_key')

def load_tickets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['support_tickets']

def convert_to_pd():
    tickets = load_tickets()
    return pd.DataFrame(tickets)

## Step 2 : load tickts as langchain documents 
def load_as_documents(file_path):
    tickets = load_tickets(file_path)
    documents = []
    for ticket in tickets:
        doc = Document(
                page_content = ticket['content'],
                metadata = {
                    'source_id': ticket['id'],
                    'title': ticket['title'],
                    'category': ticket['category'],
                    'priority': ticket['metadata']['priority'],
                    'product_area': ticket['metadata']['product_area']
                }
        )
        documents.append(doc)
    return documents

documents = load_as_documents(FILE_PATH)

## Step 3 : configure splitter 
splitter = RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=30 ,separators = ['\n\n','\n','. ',' ',''])

## Step 4: split langchain documents into chunks
chunked_documents = splitter.split_documents(documents)

## Step 5: Assign unique chunk id to each chunk metadata
for id,chunk in enumerate(chunked_documents,start=1):
     chunk.metadata['chunk_id'] = f'CHUNK-{id:03d}'

print(f'Total Chunks : {len(chunked_documents)}')
print(f'Chunked data : \n\n {chunked_documents[4]}')
print(f'Chunked data size : { len(chunked_documents[4].page_content) }')


## Step 6: Set up embedding model 
embedding_model = OpenAIEmbeddings(api_key = anek_applied_ai_key,model = 'text-embedding-3-small')

## Step 7: Embed One chunk 
"""
chunked_documents[0].page_content
        │
        │ entire chunk text
        ▼
   Embedding Model
        │
        ▼
[0.12, -0.03, 0.87, ..., 0.04]
        │
        └── 1536 numbers
"""

# sample_text = chunked_documents[1].page_content
# vector = embedding_model.embed_query(chunked_documents[0].page_content)

# print(f'Sample text :: \n {sample_text[:50]}')
# print(f'vector length :: {len(vector)}')
# print(f'First 5 value :: {vector[:5]}')

#all_vectors = embedding_model.embed_documents( [doc.page_content for doc in chunked_documents] )
# print(f'lenght of all vectos :: {len(all_vectors)}')
# print(f'each vector length :: {len(all_vectors[0])}')


# Vector Store 
CHROMA_DB_PATH = '/Users/anekkumarsingh/vector_db/chroma_db'
chunk_ids = [ chunked_document.metadata['chunk_id'] for chunked_document in chunked_documents ]

vector_store = Chroma.from_documents(
                        documents = chunked_documents,
                        embedding = embedding_model,
                        persist_directory = CHROMA_DB_PATH,  
                        collection_name = 'support_tickets',
                        ids = chunk_ids 
                    )

print(f'item indexed : {vector_store._collection.count()}')

results = vector_store.similarity_search('How do I reset my password ?',k=3)
for result in results:
    print(f'chunk -> {result.metadata['chunk_id']} , source id -> {result.metadata['source_id']} , content -> {result.page_content}')

