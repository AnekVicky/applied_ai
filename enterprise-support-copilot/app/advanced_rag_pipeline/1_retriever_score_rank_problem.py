import os 
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()
anek_applied_ai_key = os.getenv('anek_applied_ai_key')

CHROMA_DB_PATH = '/Users/anekkumarsingh/vector_db/chroma_db'

embeddings = OpenAIEmbeddings(
    api_key = anek_applied_ai_key,
    model = 'text-embedding-3-small'
)

vectorstore = Chroma(
    persist_directory = CHROMA_DB_PATH,
    embedding_function = embeddings,
    collection_name = 'support_tickets'
)
def get_questions():
    return [
        "What is AnekVicky Cloud's refund policy for Enterprise cancellations?",
        "What happens if I want to add more team members to my Enterprise account?",
        "How can I make my account more secure?"
        ]
def expected_top_source():
    return {
    "What is AnekVicky Cloud's refund policy for Enterprise cancellations?": "TCK-001",
    "What happens if I want to add more team members to my Enterprise account?": "TCK-007",
    "How can I make my account more secure?": "TCK-005"
    }
TOP_K = 3
def run_pipeline():
    questions = get_questions()
    expected_top_sources = expected_top_source()

    for question in questions:
        retrieved_chunked_results = vectorstore.similarity_search_with_score(question, k = TOP_K)
        print('='*30)
        print(f'Question : {question}')
        print('='*30)
        for rank,(retrieved_chunk,score) in enumerate(retrieved_chunked_results,start = 1):
            marker = '<---- Expected ' if retrieved_chunk.metadata['ticket_id'] == expected_top_sources[question] else ''
            print(f'rank : {rank} ,score : {score} , retrieved_chunk source : {retrieved_chunk.metadata['ticket_id']} {marker}')

if __name__ == '__main__':
    run_pipeline()


