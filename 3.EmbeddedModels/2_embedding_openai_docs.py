from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small", dimensions = 32
)

document = [
    "Delhi is the capital of India",
    "paris is the capital of France",
    "Washington DC is the capital of USA"
]

result = embedding.embed_documents(document)
print(str(result))