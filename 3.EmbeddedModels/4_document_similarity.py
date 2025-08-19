from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = "text-embedding-3-small", dimensions=300)

document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]


query = "tell me about Virat kohli"

docs_embedding = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

#print(cosine_similarity([query_embedding],docs_embedding)) # Inside cosine_similarity, we need to pass query_embedding and docs_embedding as 2 dimensional vector.

score = cosine_similarity([query_embedding],docs_embedding)[0]
#print(list(enumerate(score)))

# sort the similarity score
index , score = sorted(list(enumerate(score)),key = lambda x:x[1])[-1]

print(query)
print(document[index])
print("similarity score is : ",score)
