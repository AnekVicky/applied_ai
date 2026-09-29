import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI,OpenAIEmbeddings


load_dotenv()
anek_applied_ai_key = os.getenv('anek_applied_ai_key')

embedding_model = OpenAIEmbeddings(api_key = anek_applied_ai_key,model='text-embedding-3-small')
CHROMA_DB_PATH = '/Users/anekkumarsingh/vector_db/chroma_db'

vectorstore = Chroma(persist_directory=CHROMA_DB_PATH,
                     embedding_function= embedding_model,
                     collection_name = 'support_tickets')

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(f'connected to vectorstore : {vectorstore._collection.count()}')

llm = ChatOpenAI(api_key=anek_applied_ai_key,model='gpt-5.6-luna',max_completion_tokens=200)

#Generation matrix : 1.faithfulness 2.correctness 3.Refusal
TOP_K = 3

SYSTEM_INSTRUCTION = """ You are an expert helping assistant in Question Answer.You are given a document with format 
[Source : ticket_id-title]\npage_content  as below context . Treat title as the part of content only .
Answer the user question based on the context .Answer can be in titile or page_content.
If you dont find any answer then dont make up any answer and just say "I don't know" .

context : {context}
"""
qa_prompt = ChatPromptTemplate(
    [
        ('system', SYSTEM_INSTRUCTION),
        ('human', '{input}')
    ]
)

chain = qa_prompt | llm | StrOutputParser()

TOP_K = 3

def format_docs(docs):
    parts = []
    for doc in docs:
        ticket_id = doc.metadata['ticket_id']
        title = doc.metadata['title']
        my_chunk = f'Source : {ticket_id}-{title}\n{doc.page_content}'

        parts.append(my_chunk)
    return "\n\n".join(parts)

def llm_as_judge(generated_answer,context):
    if "I don't know".lower() in generated_answer.lower():
     return "N/A (refusal)"

    judge_prompt = f"""Context:
    {context}
    Generated answer: {generated_answer}
    Is every claim in the generated answer directly supported by the context
    above? Respond with ONLY one word: Faithful, Partial, or Unfaithful.
    """
    #judge_chain = judge_prompt | llm | StrOutputParser() 
    response = llm.invoke([HumanMessage(content=judge_prompt)])

    return response.content.strip()
def refusal_check(generated_answer,context,question_set):
    is_refused_by_llm = "I don't know".lower() in generated_answer.lower()
    should_refuse = len(question_set['expected_sources']) == 0 

    if is_refused_by_llm and should_refuse:
        return 'correct refusal'
    elif is_refused_by_llm and not should_refuse:
        return 'llm refusal but it should not'
    elif not is_refused_by_llm and should_refuse:
        return 'llm hallucinate'
    elif not is_refused_by_llm and not should_refuse:
        return 'correctly answered'

    


def run_pipeline_evaluation(TEST_SETS):
    for TEST_SET in TEST_SETS:
        print("="*30)
        user_question = TEST_SET['question']
        print(f'user_question : {user_question}')
        
        retrieved_docs = retriever.invoke(TEST_SET['question'])
        context = format_docs(retrieved_docs)
        print(f'retrived docs context :\n {context}')
        generated_answer = chain.invoke({'input':user_question,'context':context})
        print(f"llm_answer : \n\n{generated_answer}")
        print("="*30)
        faithfulness_eval = llm_as_judge(generated_answer,context)
        print(f'faithfulness_eval : {faithfulness_eval}')
        print("="*30)

        refusal_eval = refusal_check(generated_answer,context,TEST_SET)
        print(f'refusal_eval : {refusal_eval}')
        print("="*30)


    return 

#Run the eval
TEST_SETS = [
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

#RUN
run_pipeline_evaluation(TEST_SETS)  


    
