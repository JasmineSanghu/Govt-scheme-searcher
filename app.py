import os
import json
import hashlib
import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from google import genai

load_dotenv()

USERS_FILE = "users.json"
CONVERSATIONS_FILE = "conversations.json"

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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
    tools = [
        Tool(
            name="Local_Database",
            func=local_policy_search,
            description="Best for official government scheme details and eligibility from internal files."
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

Question: {input}
Thought:{agent_scratchpad}"""
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center; color: white;
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .main-header p { font-size: 0.95rem; opacity: 0.8; margin: 0.5rem 0 0 0; }
    .stTextInput > div > div > input {
        border-radius: 8px; border: 1.5px solid #e0e0e0;
        padding: 0.6rem 1rem; font-size: 0.95rem;
    }
    .stButton > button {
        border-radius: 8px; font-weight: 600;
        padding: 0.5rem 1.5rem; transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .user-badge {
        background: linear-gradient(135deg, #0f3460, #1a1a2e);
        color: white; padding: 0.5rem 1rem; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; text-align: center; margin-bottom: 1rem;
    }
    .conv-active {
        background: #e8f0fe; border-left: 3px solid #0f3460;
        border-radius: 6px; padding: 0.4rem 0.7rem; margin: 0.2rem 0;
        font-size: 0.9rem; font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

def init_session():
    for key, val in {"logged_in": False, "username": None, "current_conversation_id": None, "messages": []}.items():
        if key not in st.session_state:
            st.session_state[key] = val

def login(username, password):
    users = load_json(USERS_FILE)
    if username in users and users[username]["password"] == hash_password(password):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.messages = []
        st.session_state.current_conversation_id = None
        return True
    return False

def register(username, password):
    users = load_json(USERS_FILE)
    if username in users:
        return False
    users[username] = {"password": hash_password(password), "created": str(datetime.datetime.now())}
    save_json(USERS_FILE, users)
    return True

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.messages = []
    st.session_state.current_conversation_id = None

def get_user_conversations():
    all_convs = load_json(CONVERSATIONS_FILE)
    return all_convs.get(st.session_state.username, {})

def save_user_conversations(convs):
    all_convs = load_json(CONVERSATIONS_FILE)
    all_convs[st.session_state.username] = convs
    save_json(CONVERSATIONS_FILE, all_convs)

def new_chat():
    conv_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    convs = get_user_conversations()
    convs[conv_id] = {"title": "New Chat", "messages": [], "created": str(datetime.datetime.now())}
    save_user_conversations(convs)
    st.session_state.current_conversation_id = conv_id
    st.session_state.messages = []

def switch_conversation(conv_id):
    convs = get_user_conversations()
    st.session_state.current_conversation_id = conv_id
    st.session_state.messages = convs[conv_id]["messages"]

def show_auth_page():
    st.markdown('<div class="main-header"><h1>🏛️ GovGuide</h1><p>Streamlined Government Scheme Discovery & Policy Navigation</p></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Create Account"])
        with tab1:
            st.markdown("#### Welcome back")
            username = st.text_input("Username", key="login_user", placeholder="Enter username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
            if st.button("Login", use_container_width=True, type="primary"):
                if login(username, password):
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        with tab2:
            st.markdown("#### Create Account")
            new_user = st.text_input("Username", key="reg_user", placeholder="Choose a username")
            new_pass = st.text_input("Password", type="password", key="reg_pass", placeholder="Choose a password")
            confirm_pass = st.text_input("Confirm Password", type="password", key="reg_confirm", placeholder="Confirm password")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif len(new_user) < 3:
                    st.error("Username must be at least 3 characters")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters")
                elif register(new_user, new_pass):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already exists")

def show_main_app():
    convs = get_user_conversations()
    with st.sidebar:
        st.markdown(f'<div class="user-badge">👤 {st.session_state.username}</div>', unsafe_allow_html=True)
        if st.button(" New Chat", use_container_width=True, type="primary"):
            new_chat()
            st.rerun()
        st.divider()
        st.caption(" Previous Conversations")
        for conv_id, conv_data in reversed(list(convs.items())):
            is_active = conv_id == st.session_state.current_conversation_id
            if is_active:
                st.markdown(f'<div class="conv-active">→ {conv_data["title"]}</div>', unsafe_allow_html=True)
            else:
                if st.button(f"💬 {conv_data['title'][:35]}", key=conv_id, use_container_width=True):
                    switch_conversation(conv_id)
                    st.rerun()
        st.divider()
        if st.button(" Logout", use_container_width=True):
            logout()
            st.rerun()

    st.markdown('<div class="main-header"><h1>🏛️ GovGuide</h1><p>Streamlined Government Scheme Discovery & Policy Navigation</p></div>', unsafe_allow_html=True)

    if st.session_state.current_conversation_id is None:
        st.markdown("###  Click **New Chat** to get started")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query := st.chat_input("Ask me about any government scheme..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            conv_id = st.session_state.current_conversation_id
            convs = get_user_conversations()
            if convs[conv_id]["title"] == "New Chat":
                convs[conv_id]["title"] = user_query[:40]
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    agent_executor = load_agent()
                    response = agent_executor.invoke({"input": user_query})
                    answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            convs[conv_id]["messages"] = st.session_state.messages
            save_user_conversations(convs)
            st.rerun()

st.set_page_config(page_title="GovGuide", page_icon="🏛️", layout="wide")
apply_styles()
init_session()

if not st.session_state.logged_in:
    show_auth_page()
else:
    show_main_app()
