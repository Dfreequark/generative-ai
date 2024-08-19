from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

# loading environment variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

# loading external pdf
loader = PyPDFLoader("stories.pdf")
pdf_text =loader.load()


# chunking the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap =200)
pdf_chunk = text_splitter.split_documents(pdf_text)


# vector database
embeddings = CohereEmbeddings(
    model = "embed-english-v3.0"
)

db = FAISS.from_documents(pdf_chunk, embeddings)

# model initialize
model = ChatCohere(model="command-r-plus")

# prompt
sytem_message = (
    """ 
You are an assistant for question answring tasks.
use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.
Answer in {language} language.

\n\n
{context}

"""
)

new_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", sytem_message),
        ("human", "{input}")
    ]

)

# retriever initialize
retriever = db.as_retriever()

# retrieval chain creation
question_answer_chain = create_stuff_documents_chain(model, new_prompt)
new_rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# streamlit framework
st.title("ASK FROM PDF")
input_question= st.text_input("Enter your query: ")
input_language =st.text_input("Language use ?: ")

if input_question and input_language:
    new_response= new_rag_chain.invoke({"input":input_question, "language": input_language})
    st.write(new_response["answer"])