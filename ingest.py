import os
import requests
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
os.environ["USER_AGENT"] = "energy-rag-project"

urls = [
    "https://www.eia.gov/outlooks/steo/",
    "https://www.eia.gov/todayinenergy/",
    "https://www.caiso.com/about/news/energy-matters-blog/summer-readiness-activities-are-well-underway",
    "https://blog.gridstatus.io/caiso-beats-the-heat/",
    "https://www.caiso.com/todays-outlook",
]

print("Loading documents...")
loader = WebBaseLoader(urls)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="/Users/ruchi/Desktop/chroma_db"
)

print("Vector store saved to ./chroma_db")
print(f"Total chunks indexed: {len(chunks)}")