import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.agents import AgentExecutor, create_react_agent
from langchain.embeddings.base import Embeddings
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
import google.generativeai as genai

load_dotenv()

# --- 1. GEMINI EMBEDDINGS (same as ingest.py) ---
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

# --- 2. LOAD THE VECTOR DB ---
@st.cache_resource
def load_vector_db():
    embeddings = GeminiEmbeddings()
    return Chroma(persist_directory="./db", embedding_function=embeddings)

# --- 3. DEFINE TOOLS ---
def local_policy_search(query: str):
    """Searches the local database for government schemes."""
    docs = load_vector_db().similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

web_search = TavilySearchResults(k=3)

tools = [
    Tool(
        name="Local_Database",
        func=local_policy_search,
        description="Best for official government scheme details and eligibility from our internal files."
    ),
    Tool(
        name="Internet_Search",
        func=web_search.run,
        description="Best for finding real-time updates, news, or very recent changes in policies."
    )
]

# --- 4. INITIALIZE AGENT ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Agentic Policy AI", page_icon="🏛️")
st.title("🏛️ Agentic Government Policy Assistant")

st.sidebar.title("Configuration")
mode = st.sidebar.radio("Select Search Mode:", ["Local Schemes File", "Live Web Search (Google)"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask me about any scheme..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response = agent_executor.invoke({"input": user_query})
        answer = response["output"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
