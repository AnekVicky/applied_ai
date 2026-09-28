import json ,os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI



load_dotenv()
anek_applied_api_key = os.getenv('anek_applied_ai_key')

embedding_model = OpenAIEmbeddings(
    api_key = anek_applied_api_key,
    model = 'text-embedding-3-small'
)
llm = ChatOpenAI(
    api_key = anek_applied_api_key,
    model= 'gpt-5.6-luna',
    max_tokens = 200
)
CHROMA_DB_PATH = '/Users/anekkumarsingh/vector_db/chroma_db'
vectorstore = Chroma(
    persist_directory= CHROMA_DB_PATH,
    embedding_function = embedding_model,
    collection_name = 'support_tickets'
)
retriever = vectorstore.as_retriever()
print(f'connected to vectorstore ,total count : {vectorstore._collection.count()}')

SYSTEM_INSTRUCTION = """ You are an expert helping assistant in Question Answer.You are given a document with format 
[Source : ticket_id-title]\npage_content  as below context . Treat title as the part of content only .
Answer the user question based on the context .Answer can be in titile or page_content.
If you dont find any answer then dont make up any answer and just say "I don't know" .

context : {context}
"""
# qa_prompt = ChatPromptTemplate(
#     ('system',SYSTEM_INSTRUCTION),('human','{input}')
# )

TOP_K = 3
TEST_SET = [
{
"question": "What is Dandes Cloud's refund policy for Enterprise cancellations?",
"expected_sources": ["TCK-001"]
},
{
"question": "How do I reset my password?",
"expected_sources": ["TCK-002"]
},
{
"question": "What happens if I exceed the API rate limit?",
"expected_sources": ["TCK-003", "TCK-028"]
},
{
"question": "How do I set up two-factor authentication?",
"expected_sources": ["TCK-005"]
},
{
"question": "What is the capital of India?",
"expected_sources": [] # NOT in knowledge base - should refuse
}
]
    

def format_docs(docs):
    parts = []
    for doc in docs:
        ticket_id = doc.metadata['ticket_id']
        title = doc.metadata['title']
        my_chunk = f'[Source : {ticket_id}-{title}]\n{doc.page_content}'
        parts.append(my_chunk)

    return '\n\n'.join(parts)
    

def run_evaluation_pipeline():

    for test in TEST_SET:
        docs = retriever.invoke(test['question'],k = TOP_K)
        retrieved_sources = set([doc.metadata['ticket_id'] for doc in docs])
        expected_sources = set(test['expected_sources'])


        true_positive = len(retrieved_sources & expected_sources) 

        precision = true_positive / len(retrieved_sources) if retrieved_sources else 0 
        recall_k = true_positive / len(expected_sources) if expected_sources else 0
        f1 = 0 if (precision + recall_k) == 0 else 2 * (precision * recall_k ) / (precision + recall_k)

        print(f'user question : {test['question']}')
        print(f'retrieved sources :: {retrieved_sources}')
        print(f'expected_sources :: {expected_sources}')
        print(f'precision : {precision}, recall_k : {recall_k} , f1_score : {f1}')

 

        

run_evaluation_pipeline()

