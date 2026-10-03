import os 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

load_dotenv()
anek_applied_ai_key = os.getenv('anek_applied_ai_key')

CHROMA_DB_PATH = '/Users/anekkumarsingh/vector_db/chroma_db'
embedding_model = OpenAIEmbeddings(
                    api_key = anek_applied_ai_key,
                    model = 'text-embedding-3-small'
                    )
vectorstore = Chroma(
    persist_directory = CHROMA_DB_PATH,
    embedding_function = embedding_model,
    collection_name = 'support_tickets'
)
print(f'vectostore connected ,count : {vectorstore._collection.count()}')

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print(f'+++++++++++++++++ cross_encoder is loaded ++++++++++++++++++++')

llm = ChatOpenAI(api_key = anek_applied_ai_key,model='gpt-5.6-luna',max_completion_tokens=200)

SYSTEM_INSTRUCTION = """ You are an expert helping cloud assistant in Question Answer.You are given a document with format 
[Source : ticket_id-title]\npage_content  as below context . Treat title as the part of content only .
Answer the user question based on the context .Answer can be in titile or page_content.
If you dont find any answer then dont make up any answer and just say "I don't know" .

context : {context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system',SYSTEM_INSTRUCTION),('human','{input}')
    ]

)
print("Prompt variables:", qa_prompt.input_variables)
chain = qa_prompt | llm | StrOutputParser()

#===========================================================================
# PROBLEM: Keyword-Based Auto-Detection (Method #2, from Metadata Filtering)
#===========================================================================
KEYWORD_MAP = {
            "billing": ["billing", "refund", "payment", "subscription"],
            "security": ["security", "password", "2fa", "authentication"],
            "onboarding": ["onboarding", "setup", "getting started", "signup"],
            "technical": ["api", "error code", "rate limit", "bug"],
}
#=================================================
# Test Questions — Deliberately Indirect Phrasing
#=================================================
def get_test_questions():
    return  [
             "Any issues with my login?", # obviously security, no keyword hit
             "I got charged twice for the same invoice", # obviously billing, no keyword hit
             "Is there an outage right now?", # obviously technical, no keyword hit
             "Any security issue do I need to take care ?"
        ]

def keyword_based_auto_detection(question):
    question_lower = question.lower()
    found_category = None

    for category,keywords in KEYWORD_MAP.items():
         for keyword in keywords:
               if keyword in question:
                   found_category = category
                   
    return found_category
             

TOP_K = 3
def format_docs(retrieved_chunked_docs):
    parts=[]
    for chunk in retrieved_chunked_docs:
        ticket_id = chunk.metadata['ticket_id']
        title = chunk.metadata['title']
        my_chunk = f'Source : {ticket_id}-{title}\n{chunk.page_content}'
        parts.append(my_chunk)
    return '\n\n'.join(parts)


def run_pipeline():
    questions = get_test_questions()

    for question in questions:
        detected_category = keyword_based_auto_detection(question)
        search_filter = {'category' : detected_category} if detected_category else None

        retrieved_chunked_docs = vectorstore.similarity_search(question,k = TOP_K,filter = search_filter)
        context = format_docs(retrieved_chunked_docs)
  
        generated_answer = chain.invoke({'context':context,'input':question})

        print(f'='*30)
        print(f'detected_category : {detected_category}')
        print(f'Question : {question}')
        print(f'='*30)
        print(f'Answer :\n\n : {generated_answer}')
        print(f'='*30)

    return 



if __name__ == '__main__':
    run_pipeline()