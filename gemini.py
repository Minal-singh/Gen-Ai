import os

from dotenv import load_dotenv
import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("LangChain Gemini 2.0 Chat")
input_text = st.text_input("Enter your question here:")

output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant that can answer questions."),
        ("user", "Question: {question}"),
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=1000,
    timeout=2,
    max_retries=2,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
