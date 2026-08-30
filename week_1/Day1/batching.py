from langchain_groq import ChatGroq
from groq import Groq
import os 
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(
    model="groq/compound-mini",
    temperature=0.7,
    max_tokens=1000
    )

responses = model.batch([
    'what is Agentic AI',
    'what is Shirk?'],
    config={
        "max_concurrency":5  # limit to the 5 parallel calls
    }

)

for response in responses:
    print(response.text)
    print('------------------------------')
