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


template = load_prompt("template.json")

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
    chain = template | model
    result = chain.invoke(
        {
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input
        }
    )
    st.success(result.content)