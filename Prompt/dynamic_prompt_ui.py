from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt

from dotenv import load_dotenv

import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(repo_id = "mistralai/Mistral-7B-Instruct-v0.2", task = "text-generation")

model = ChatHuggingFace(llm = llm)

st.header("Research Tool")

paper_input = st.selectbox("Select the research paper name",options=["Machine Learning","Deep Learning","Artificial Intellingence"])

style_input = st.selectbox("Select the style of writing",options=["Beginner friendly","Technical","Mathematical"])

length_input = st.selectbox("Select the length of the summary",options=["Short","Medium","Long"])

# template
# template = PromptTemplate(
#     template = """
#            Please summarize the research paper titled "{paper_input}" with the following specifications:
# Explanation Style: {style_input}
# Explanation Length: {length_input}

# 1. Mathematical Details:
#    - Include relevant mathematical equations if present in the paper.
#    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

# 2. Analogies:
#    - Use relatable analogies to simplify complex ideas.

# If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

# Ensure the summary is clear, accurate, and aligned with the provided style and length.
# """, input_variables=["paper_input","style_input","length_input"],
# validate_template = True
# )
# validate template to ensure all input variables are present and correctly formatted. This helps catch errors early in the prompt construction process.

# We can load the prompt template from any other file also and use it here.
template = load_prompt("template.json")

# fill the placeholder
prompt = template.invoke(
    {
        "paper_input":paper_input,
        "style_input":style_input,
        "length_input":length_input
    }
)

# Custom CSS targeting Streamlit button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: white;
        color: red;
        border: 2px solid red;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: bold;
        
    }
    div.stButton > button:first-child:hover {
        background-color: red;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.success(result.content)