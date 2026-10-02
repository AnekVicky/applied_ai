import os 
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder


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
#cross_encoder = CrossEncoder('cross_encoder/ms-macro-MiniLM-L-6-v2')
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print('++++++++   cross_encoder model loaded +++++++++++++++++++++')

def get_questions():
    return [
        "What is Dandes Cloud's refund policy for Enterprise cancellations?",
        "What happens if I want to add more team members to my Enterprise account?",
        "How can I make my account more secure?"
        ]
def expected_top_source():
    return {
    "What is Dandes Cloud's refund policy for Enterprise cancellations?": "TCK-001",
    "What happens if I want to add more team members to my Enterprise account?": "TCK-007",
    "How can I make my account more secure?": "TCK-005"
    }

TOP_K = 3
RETRIEVED_K = 10

def run_pipeline():
    questions = get_questions()
    expected_top_sources = expected_top_source()

    for question in questions:
        retrieved_chunked_results = vectorstore.similarity_search(question, k = RETRIEVED_K)
        print('='*30)
        print(f'Question : {question}')
        print('='*30)

        # Use Encoder for reranking  NOTE here : vectorstore.similarity_search or else you will have issues in result.page_content
        question_result_pair = [(question,result.page_content) for result in retrieved_chunked_results]
        #print(f'question_result_pair : {question_result_pair}')
        scores = cross_encoder.predict(question_result_pair)

        ranked = sorted(zip(retrieved_chunked_results,scores),key = lambda x : x[1] ,reverse=True)

        for rank,(retrieved_chunk,score) in enumerate(ranked[:TOP_K],start = 1):
            marker = '<---- Expected ' if retrieved_chunk.metadata['ticket_id'] == expected_top_sources[question] else ''
            print(f'rank : {rank} ,score : {score} , retrieved_chunk source : {retrieved_chunk.metadata['ticket_id']} {marker}')

if __name__ == '__main__':
    run_pipeline()


