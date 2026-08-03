from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma(
    collection_name="medical_textbooks",
    persist_directory="./medical_langchain_db",
    embedding_function=embeddings
)

# Get documents
docs = vector_store.get()
print(docs)  # This will show your actual data
