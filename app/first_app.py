from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to user queries. Say at the end of the answer : 'Thank You for asking! Have a great day!'"),
        ("user", "Question: {question}")
    ]
)

prompt_hindi = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to user queries in hindi language"),
        ("user", "Question: {question}")
    ]
)

prompt_language = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to user queries in {language} language"),
        ("user", "Question: {question}")
    ]
)


model = ChatCohere(model="command-r-plus")
ourput_parser = StrOutputParser()

# chain creation
chain = prompt_language | model | ourput_parser

# streamlit framework
st.title("OUR ASSISTANT")
input_text = st.text_input("Enter your question here :")
input_language = st.text_input("In what language do you want your answer :")

if input_text and input_language :
    st.write(chain.invoke({ "language": input_language,"question": input_text}))