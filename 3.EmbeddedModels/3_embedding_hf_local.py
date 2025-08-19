from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# text = "Delhi is the capital of India."
# vector = embedding.embed_query(text)

documents = [
    "Delhi is the capital of India",
    "paris is the capital of France",
    "Washington DC is the capital of USA"
]
vector = embedding.embed_documents(documents)
print(str(vector))