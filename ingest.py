import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from google import genai

load_dotenv()

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

def create_vector_db():
    loader = TextLoader("schemes.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = GeminiEmbeddings()
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./db"
    )
    print(f"Successfully created vector store with {len(chunks)} chunks.")

if __name__ == "__main__":
    create_vector_db()
