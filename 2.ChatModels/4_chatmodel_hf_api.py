from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # When we want to use HuggingFace API then we will import HuggingFaceEndpoint.

from dotenv import load_dotenv

load_dotenv()

# Configuring llm by using HuggingFaceEndpoint.

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

#  # repo_id is the model id from HuggingFace, task is the type of task we want to perform like text-generation, image-generation, etc.

# we will get llm from HuggingFaceEndpoint.
model = ChatHuggingFace(llm=llm)



result = model.invoke("What is the capital of India ?")

print(result.content)