import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def create_vector_db():
    # 1. Load data
    loader = TextLoader("schemes.txt", encoding="utf-8")
    documents = loader.load()

    # 2. Split (The NLP/ML part)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # 3. Embed and Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./db"
    )
    print(f"Successfully created vector store with {len(chunks)} chunks.")

if __name__ == "__main__":
    create_vector_db()
