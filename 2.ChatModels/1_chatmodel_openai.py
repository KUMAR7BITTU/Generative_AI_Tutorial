# We will use here chatmodel interface and communicate with OpenAI gpt-4 model.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv() 

# value of temperature will be from 0 to 2. It tells how much random or creative response you want from the model.
# max_completion_tokens helps us to set how many tokens we want in the output. # tokens are roughly words but they are not exactly words.
model = ChatOpenAI(model="gpt-4", temperature=0, max_completion_tokens=10)
#result = model.invoke("What is the capital of India?")
result = model.invoke("Write a poem on cricket.")
#print(result)
print(result.content) # It will print only the answer part.