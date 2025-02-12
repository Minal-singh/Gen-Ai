from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant that can answer questions."),
        ("user", "Question: {question}"),
    ]
)

st.title("LangChain OpenAI Chat")
input_text = st.text_input("Enter your question here:")

output_parser = StrOutputParser()

repo_id = "deepseek-ai/DeepSeek-R1"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    max_new_tokens=500,
)

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
