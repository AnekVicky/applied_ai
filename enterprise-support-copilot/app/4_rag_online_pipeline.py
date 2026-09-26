import json ,os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage,HumanMessage
#git push git@github.com:AnekVicky/applied_ai.git main


load_dotenv()
anek_applied_ai_key = os.getenv('anek_applied_ai_key')

# Vector Store 
CHROMA_DB_PATH = '/Users/anekkumarsingh/vector_db/chroma_db'

embedding_model = OpenAIEmbeddings(api_key = anek_applied_ai_key,model = 'text-embedding-3-small')

llm = ChatOpenAI(
    api_key = anek_applied_ai_key,
    model = 'gpt-5.6-luna',
    max_tokens = 200
)

def format_docs(retrieved_docs):
    parts = []
    for doc in retrieved_docs:
       ticket_id = doc.metadata['ticket_id']
       title = doc.metadata['title']
       my_chunks = f'[Source : {ticket_id}-{title}]\n{doc.page_content} '
       parts.append(my_chunks)

    return "\n\n".join(parts)



vector_store = Chroma(persist_directory=CHROMA_DB_PATH,
                      embedding_function = embedding_model,
                      collection_name = 'support_tickets')
print(f'vector_store connected : {vector_store._collection.count()}')

SYSTEM_INSTRUCTION = """ You are an expert in Question Answer.You are given a document with format 
[Source : {ticket_id}-{title}]\n{doc.page_content} . Treat title as the part of content only .
Answer the user question based on the above context .Answer can be in titile or page_content.
If you dont find any answer then dont make up any answer and just say I dont know .
"""
TOP_K = 3

def run_pipeline(query):

    query_vector = embedding_model.embed_query(query)
    print(f'use query vector : {query_vector[:5]}')
    retrieved_docs = vector_store.similarity_search_by_vector(query_vector,k = TOP_K)
    
    for doc in retrieved_docs:
     print(f'chunk_id -> {doc.metadata['chunk_id']} , ticket_id -> {doc.metadata['ticket_id']} ,page_content -> {doc.page_content[:10]}')

    context = format_docs(retrieved_docs)
    augmented_prompt = f"""
    Context : {context} 

    Question : {query}"""

    message = [
       SystemMessage(content = SYSTEM_INSTRUCTION),
       HumanMessage(content = augmented_prompt)
    ]

    response = llm.invoke(message)
    print(f'Generated answer :: {response.content}')


query = "How do I reset my password?"
run_pipeline(query)
