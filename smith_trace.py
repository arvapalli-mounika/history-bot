# import os

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = "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"

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import LangChainTracer
from dotenv import load_dotenv

load_dotenv(override=True)
tracer = LangChainTracer(project_name="HistoryBot")


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
    ("user", "Question: {question}\nContext: {context}")
])
model = ChatOllama(model="llama3.2:latest", temperature=0.7)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

question = "Can you give one line answer about LangSmith used for?"
context = "I'm a GenAI learner in beginner level."
chain.invoke({"question": question, "context": context},config={"callbacks": [tracer]})