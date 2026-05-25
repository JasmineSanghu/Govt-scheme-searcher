import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from google import genai

load_dotenv()

# --- 1. EMBEDDINGS ---
class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def embed_documents(self, texts):
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text):
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=[text],
        )
        return result.embeddings[0].values

# --- 2. LOAD VECTOR DB ---
@st.cache_resource
def load_vector_db():
    embeddings = GeminiEmbeddings()
    return Chroma(persist_directory="./db", embedding_function=embeddings)

# --- 3. TOOLS ---
def local_policy_search(query: str):
    docs = load_vector_db().similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

web_search = TavilySea
