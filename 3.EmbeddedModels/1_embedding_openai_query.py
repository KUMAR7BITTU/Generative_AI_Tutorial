from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

# If we increase the size of vector by increasing the dimensions, then more contextual meaning of the text will be captured. 
# more the size of vector, more the price .
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small", dimensions=32
)

result = embedding.embed_query("Delhi is the capital of India.") # this will generate a vector of 32 dimensions for the given text.

print(str(result))