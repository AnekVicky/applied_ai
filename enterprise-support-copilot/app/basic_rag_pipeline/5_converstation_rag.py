import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma 
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
anek_applied_ai_key =  os.getenv('anek_applied_ai_key')

llm = ChatOpenAI(
    api_key = anek_applied_ai_key,
    model = 'gpt-5.6-luna',
    max_tokens = 200
)

embedding_model = OpenAIEmbeddings(api_key = anek_applied_ai_key,model = 'text-embedding-3-small')

CHROMA_DB_PATH = '/Users/anekkumarsingh/vector_db/chroma_db'
vector_store = Chroma(persist_directory = CHROMA_DB_PATH,
                      embedding_function = embedding_model,
                      collection_name = 'support_tickets'
                      )
retriever = vector_store.as_retriever()
print(f'vector_store collection_name ::-> count :: {vector_store._collection.count()}  ')

#====================================================
# reformulation 
#====================================================
# ====================================================
# Reformulation
# ====================================================

REFORMULATION_INSTRUCTION = """
You are a query reformulator for a conversational RAG system.

Rewrite the CURRENT USER QUESTION into a standalone question that can be
understood without the conversation history.

Rules:

1. Preserve the current user's intent.
2. Use the conversation history to resolve references such as:
   - it
   - this
   - that
   - they
   - them
   - what about
   - how about
3. If the current question changes an entity, use the new entity.

Example:

Previous:
User: What is the API rate limit for the Enterprise plan?
Assistant: The Enterprise plan allows 1,000 API requests per minute.

Current:
what about the pro plan?

Output:
What is the API rate limit for the Pro plan?

4. Do not answer the question.
5. Return ONLY the standalone question.
6. If the current question is already standalone, return it unchanged.
"""

CONTEXTUALIZE_PROMPT = ChatPromptTemplate(
    [
        ("system", REFORMULATION_INSTRUCTION),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

reformulate_chain = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

CONTEXTUALIZE_PROMPT = ChatPromptTemplate(
    [
        ('system',REFORMULATION_INSTRUCTION),
          MessagesPlaceholder('chat_history'),
        ('human','{input}')
      
    ]
)
reformulate_chain = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()


#====================================================
# qa prompt + generation  
#====================================================

SYSTEM_INSTRUCTION = """ You are an expert helping assistant in Question Answer.You are given a document with format 
[Source : ticket_id-title]\npage_content  as below context . Treat title as the part of content only .
Answer the user question based on the context .Answer can be in titile or page_content.
If you dont find any answer then dont make up any answer and just say "I don't know" .

context : {context}
"""
TOP_K = 3
qa_prompt = ChatPromptTemplate(
    [
        ('system',SYSTEM_INSTRUCTION),
        MessagesPlaceholder('chat_history'),
         ('human','{input}')
    ]
)

qa_chain = qa_prompt | llm | StrOutputParser()

chat_history = []
def format_docs(retrieved_docs):
    parts = []

    for doc in retrieved_docs:
        ticket_id = doc.metadata['ticket_id']
        title = doc.metadata['title']
        my_chunk = f'Source : {ticket_id}-{title}\n{doc.page_content}'
        parts.append(my_chunk)

    return "\n\n".join(parts)
# rag pipeline

def chat(user_message):
    
    # ============================================
    # 1. Reformulate
    # ============================================

    if len(chat_history) == 0:
        standalone_message = user_message
    else:
        standalone_message = reformulate_chain.invoke({
            'input':user_message,
            'chat_history':chat_history
        })

    # ============================================
    # 2. Retrieve
    # ============================================
    retrieved_docs = retriever.invoke(standalone_message,k = TOP_K)
    context = format_docs(retrieved_docs)
   
    # ============================================
    # 3. Generate answer
    # ============================================
    answer = qa_chain.invoke({
                            'input':standalone_message,
                            'chat_history':chat_history,
                            'context':context
                        })

    # ============================================
    # 4. Update history AFTER the turn
    # ============================================
    chat_history.append(HumanMessage(content = user_message))
    chat_history.append(AIMessage(content = answer))

    sources = [doc.metadata['ticket_id'] for doc in retrieved_docs]
    
    return {
        'query' : user_message,
        'standalone_message':standalone_message,
        'sources' : sources,
        'answer':answer
    }



# ============================================
# online question
# ============================================
questions = [
    'What is API Rate limit for the enterprise plan?',
    'what about the pro plan ?',
    'And what happens if I exceed it ?',
    'how should my application handle that error ?',
]

for id,question in enumerate(questions,start=1):
    result = chat(question)
    print("="*30)
    print(f'Turn - {id}')
    print(f'user query : {result['query']}')
    print(f'standalone_message : {result['standalone_message']}')
    print(f'sources : {result['sources']}')
    print(f'answer: {result['answer']}')
    print("="*30)


