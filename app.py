import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# --- 1. LOAD THE ML ENGINE (ChromaDB) ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = Chroma(persist_directory="./db", embedding_function=embeddings)

# --- 2. DEFINE THE TOOLS (Agentic Logic) ---
def local_policy_search(query: str):
    """Searches the local database for government schemes."""
    docs = vector_db.similarity_search(query, k=3)
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

# --- 3. INITIALIZE THE AGENT ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
prompt = hub.pull("hwchase17/react")  # The logic blueprint for reasoning
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Agentic Policy AI", page_icon="🏛️")
st.title("🏛️ Agentic Government Policy Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_query := st.chat_input("Ask me about any scheme..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # The AI now decides which tool to use!
        response = agent_executor.invoke({"input": user_query})
        answer = response["output"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
