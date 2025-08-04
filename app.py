import streamlit as st
import os
from langchain_groq import ChatGroq 
from langchain_community.embeddings import OllamaEmbeddings #uses ollama embeddinga to generate vector represntation of text
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #used for vector storage
from langchain_community.document_loaders import PyPDFLoader#used to load pdf

from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

llm=ChatGroq(api_key=os.getenv("GROQ_API_KEY"),model_name="Llama3-8b-8192")

embeddings=HuggingFaceBgeEmbeddings()


loader=PyPDFLoader(r"C:\Python_Project\feedback_bot\Cravbook.pdf")
docs=loader.load()
text_spliter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_split=text_spliter.split_documents(docs)
vectors=FAISS.from_documents(docs_split,embeddings)
retriever=vectors.as_retriever()

st.set_page_config(layout="wide")
st.title("Hello!! I am The CraveBot.")

prompt = ChatPromptTemplate.from_template(
    """You are an intelligent ChatBot and your name is CravBot.You are assisting customers at our company Crav.
Answers question regarding crave from the context provided.
dont include any text in your answers that shows you are using some context start with "at crav" or "we  at crav"
if a user ask for general queries like, order status, cook availability, cuisine options generate random answers like you are using some records to get the answers start with let me check or something similar in meaning
remember your generated answers for any follow ups user asks for that conversation
dont repeat same type of response everytime
if dont know about any information from the context tell the customer to visit  https://thecrav.com/
after every response ask for feedback and use that feedback for future conversation

Question: {input}
Context: {context}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

user_query = st.chat_input("Tell me what are you craving about")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)    

if user_query:
    context = {
        "input": user_query,
    }
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append(("user", user_query))


    response = retrieval_chain.invoke(context)
    answer = response.get("answer")

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.chat_history.append(("assistant", answer))    

