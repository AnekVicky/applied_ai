import os 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")

if openai_api_key:
    print("OpenAI API key loaded successfully.")
else:
    print('Failed to load open api key')


openai_client = OpenAI(api_key=openai_api_key)

system_prompt = "You are a helpful assistant that provides understandable technical answers"
model = "gpt-4o-mini"
max_tokens = 100
user_prompt = "Explain the concept of F1 score in ML."

openai_response = openai_client.responses.create(
    model=model,
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    max_tokens=max_tokens
)   
openai_response_text = openai_response.choices[0].message.content
print("OpenAI Response:")
print(openai_response_text)