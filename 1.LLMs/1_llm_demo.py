# in langchain_openai all the codes are written how langchain will communicate with OpenAI API.
from langchain_openai import OpenAI

# load environment variable from .env file
from dotenv import load_dotenv

load_dotenv() # Invoke this function

# make an object of OpenAI class and pass which model you want to use.
llm = OpenAI(model="gpt-3.5-turbo-instruct")

# With the help of this invoke method we can communicate with this gpt model and give the prompt here.
#  This invoke method will hit our model and pass the prompt to that model.
# Our model will process this prompt and return an response . It is stored in result.
result = llm.invoke("What is the capital of India?")

print(result)