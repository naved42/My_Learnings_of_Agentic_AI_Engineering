from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()


model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=1000
    )


for chunk in model.stream("What is Agentic AI?"):
    print(chunk.text,end="",flush=True)