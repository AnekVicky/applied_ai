import os 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from pydantic import BaseModel,Field
from typing import Optional,Literal

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

#=================================================
# Test Questions — Deliberately Indirect Phrasing
#=================================================
def get_test_questions():
    return  [
             "Any issues with my login?", # obviously security, no keyword hit
             "I got charged twice for the same invoice", # obviously billing, no keyword hit
             "Is there an outage right now?", # obviously technical, no keyword hit
             "Any security issue do I need to take care ?",
             "what is the capital of india ?"
        ]

def get_intent_based_routing(question):
    class RouteDecision(BaseModel):
        needs_retrieval : bool = Field(
                description="True if the question is about Cloud's product, "
                            "billing, account, or technical support. False if it is "
                            "small talk, loose-talk, out-of-scope, or general knowledge unrelated "
                            "to Cloud's prodcut.")
        category : Optional[
            Literal['billing','security','login','api','onboarding','performance','account']] = Field(
                default = None,
                description =  "single best-matching support category for this "
                                "question, from the exact list given. Only set this if "
                                "needs_retrieval is True. Leave it unset 'None' if the "
                                "question needs retrieval but doesn't clearly fit any "
                                "single category — in that case we'll search without a "
                                "category filter.")

    router_llm = llm.with_structured_output(RouteDecision)

    ROUTER_INSTRUCTION = """You are a routing classifier for AnekVicky Cloud's support assistant.
                            Given a user's question, decide:
                            1. Whether it needs a search of the AnekVicky Cloud knowledge base (needs_retrieval)
                            2. If it does, which category it best belongs to (category)
                            Categories available: billing, api, onboarding, security, performance, account, login.
                            Read the INTENT behind the question, not just keywords. 
                           
                             For example:
                            "any issues with my login?" is a login/security question even though it
                            never says those words directly.
                            If the question needs retrieval but doesn't clearly match one category,
                            leave category unset rather than guessing."""

    router_prompt =  ChatPromptTemplate.from_messages([
        ('system',ROUTER_INSTRUCTION),('human','{input}')
    ])

    router_chain = router_prompt | router_llm
    decision = router_chain.invoke({'input': question})
    print(f'decision : {decision}')

    return decision



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
        decision = get_intent_based_routing(question)
        if not decision.needs_retrieval:
            return "No Information Found !!!"

        search_filter = {'category' : decision.category} if decision.category else None

        retrieved_chunked_docs = vectorstore.similarity_search(question,k = TOP_K,filter = search_filter)
        context = format_docs(retrieved_chunked_docs)
  
        generated_answer = chain.invoke({'context':context,'input':question})

        print(f'='*30)
        print(f'detected_category : {decision.category}')
        print(f'Question : {question}')
        print(f'='*30)
        print(f'Answer :\n\n : {generated_answer}')
        print(f'='*30)

    return 



if __name__ == '__main__':
    run_pipeline()