import os
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from google import genai

USERS_FILE = "users.json"
HISTORY_DIR = "user_histories"

os.makedirs(HISTORY_DIR, exist_ok=True)

# ---------- User Management ----------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = hash_password(password)
    save_users(users)
    return True, "Account created successfully!"

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "Username not found."
    if users[username] != hash_password(password):
        return False, "Incorrect password."
    return True, "Login successful!"

# ---------- History Management ----------
def get_history_file(username):
    return os.path.join(HISTORY_DIR, f"{username}.json")

def load_user_history(username):
    path = get_history_file(username)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_history(username, history):
    with open(get_history_file(username), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ---------- AI Setup ----------
class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    def embed_documents(self, texts):
        result = self.client.models.embed_content(model="gemini-embedding-001", contents=texts)
        return [e.values for e in result.embeddings]
    def embed_query(self, text):
        result = self.client.models.embed_content(model="gemini-embedding-001", contents=[text])
        return result.embeddings[0].values

@st.cache_resource
def load_vector_db():
    embeddings = GeminiEmbeddings()
    return Chroma(persist_directory="./db", embedding_function=embeddings)

def local_policy_search(query: str):
    docs = load_vector_db().similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

@st.cache_resource
def load_agent():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    tavily_search = TavilySearchResults(max_results=3)
    tools = [
        Tool(
            name="Local_Database",
            func=local_policy_search,
            description="Best for official government scheme details and eligibility from internal files."
        ),
        Tool(
            name="Web_Search",
            func=tavily_search.run,
            description="Use this to search the internet for latest government scheme details, news, and updates not found in local database."
        )
    ]
    template = """Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Previous searches for context: {previous_searches}
Question: {input}
Thought:{agent_scratchpad}"""
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# ---------- Page Config ----------
st.set_page_config(page_title="Govt Policy Assistant", page_icon="🏛️")

# ---------- Login/Register UI ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.title("🏛️ Government Policy Assistant")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            success, msg = login_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.messages = []
                st.session_state.search_history = load_user_history(username)
                st.rerun()
            else:
                st.error(msg)

    with tab2:
        st.subheader("Create Account")
        new_user = st.text_input("Choose Username", key="reg_user")
        new_pass = st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("Register"):
            success, msg = register_user(new_user, new_pass)
            if success:
                st.success(msg + " Please login.")
            else:
                st.error(msg)

else:
    # ---------- Main App ----------
    st.title("🏛️ Agentic Government Policy Assistant")

    # Sidebar
    with st.sidebar:
        st.markdown(f"👤 Logged in as **{st.session_state.username}**")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.messages = []
            st.session_state.search_history = []
            st.rerun()

        st.divider()
        st.header("🕘 Search History")
        if st.session_state.search_history:
            if st.button("🗑️ Clear History"):
                st.session_state.search_history = []
                save_user_history(st.session_state.username, [])
                st.rerun()
            for item in reversed(st.session_state.search_history[-20:]):
                st.markdown(f"**{item['time']}**")
                st.markdown(f"🔍 {item['query']}")
                st.divider()
        else:
            st.info("No search history yet.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask me about any government scheme..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            agent_executor = load_agent()
            recent = st.session_state.search_history[-5:]
            previous_searches = ", ".join([h["query"] for h in recent]) if recent else "None"
            response = agent_executor.invoke({
                "input": user_query,
                "previous_searches": previous_searches
            })
            answer = response["output"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            entry = {
                "query": user_query,
                "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
            }
            st.session_state.search_history.append(entry)
            save_user_history(st.session_state.username, st.session_state.search_history)
