import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# 1. API KEY SETUP
os.environ["GOOGLE_API_KEY"] = "AIzaSyDD8UfglBh_IFdSkN3Rtucl-lGv_uzYtmY"

st.set_page_config(page_title="Govt Scheme Helper", layout="centered")
st.title("🇮🇳 Govt Scheme Assistant")

# --- Initialize Application Mode State ---
if "mode" not in st.session_state:
    st.session_state.mode = "Local Schemes File"

# Sidebar selector to toggle modes smoothly
st.sidebar.title("Configuration")
st.session_state.mode = st.sidebar.radio(
    "Select Search Mode:",
    ["Local Schemes File", "Live Web Search (Google)"]
)

@st.cache_resource
def load_data():
    # Safeguard check to ensure local context file exists
    if not os.path.exists("schemes.txt"):
        # Creates a placeholder file if it is missing
        with open("schemes.txt", "w", encoding="utf-8") as f:
            f.write("Coir Udyami Yojana: Modenizes the coir industry and offers credit-linked subsidies.")
            
    with open("schemes.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

# --- CORE EXECUTION PIPELINE ---
try:
    user_query = st.text_input("Ask me about any government scheme:")

    if user_query:
        # Initialize the underlying Language Model
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

        if st.session_state.mode == "Local Schemes File":
            # Execution Route A: Standard Local Vector Search (RAG)
            db = load_data()
            search_results = db.similarity_search(user_query, k=2)
            context = "\n".join([res.page_content for res in search_results])
            
            prompt = f"Using this info: {context}, answer: {user_query}. Be helpful."
            st.write("📂 *Searching your local schemes database...*")
            response = llm.invoke(prompt)

        else:
            # Execution Route B: Live Web Search Grounding Agent
            # Attaching Google Search tool natively to the Gemini pipeline
            llm_with_search = llm.bind_tools([{"google_search": {}}])
            
            prompt = f"You are an expert Indian Government Assistant. Use the live google search tool to discover the latest information on: {user_query}. Provide a structured layout and list source URLs."
            st.write("🌐 *Searching live Indian Government portals via Google...*")
            response = llm_with_search.invoke(prompt)

        # --- REUSABLE PARSING BLOCK ---
        # Safely extract clean string values out of raw model payload structures
        if isinstance(response.content, list) and len(response.content) > 0:
            clean_text = response.content[0].get("text", "")
        elif isinstance(response.content, dict):
            clean_text = response.content.get("text", "")
        else:
            clean_text = response.content

        st.write("### Answer:")
        st.markdown(clean_text)
        
except Exception as e:
    st.error(f"Execution Error encountered: {e}")
