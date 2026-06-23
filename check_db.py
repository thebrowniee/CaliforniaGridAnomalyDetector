import chromadb

client = chromadb.PersistentClient(path="/Users/ruchi/Desktop/chroma_db")
collection = client.list_collections()
print(collection)