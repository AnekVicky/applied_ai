import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage

load_dotenv()
anek_applied_ai_key =  os.getenv('anek_applied_ai_key')

llm = ChatOpenAI(
    api_key = anek_applied_ai_key,
    model = 'gpt-5.6-luna',
    max_tokens = 200
)

embedding_model = OpenAIEmbeddings(api_key = anek_applied_ai_key,model = 'text-embedding-3-small')


