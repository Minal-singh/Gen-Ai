import os
from dotenv import load_dotenv
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0, max_tokens=1000, timeout=2, max_retries=2
)

st.set_page_config(layout="wide")
st.title("Document Q&A using LangChain and Gemini 2.0 Flash")

prompt = ChatPromptTemplate.from_template(
    """
Answer the following question based on the context provided.
<context>
{context}
<context>
Questions: {input}
"""
)


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
        )
        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(st.session_state.docs)
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )


if uploaded_files := st.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
):
    st.session_state.docs = []
    for uploaded_file in uploaded_files:
        if uploaded_file.size > 10 * 1024 * 1024:  # 10 MB limit
            st.error(f"File {uploaded_file.name} exceeds the 10MB size limit.")
            continue
        with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    loader = PyPDFDirectoryLoader("tempDir/")
    st.session_state.docs.extend(loader.load())
    if st.button("Create Vector Store"):
        vector_embedding()
        st.write("Vector Store Created")

prompt1 = st.text_input("Enter your question here:")
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": prompt1})
    st.write(response["answer"])
