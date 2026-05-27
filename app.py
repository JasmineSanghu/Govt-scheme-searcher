import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from google import genai

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

st.set_page_config(page_title="Govt Policy Assistant", page_icon="🏛️")
st.title("🏛️ Agentic Government Policy Assistant")

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
        response = agent_executor.invoke({"input": user_query})
        answer = response["output"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
