"""
BAYMAX Medical Retrieval Tool
Handles all medical knowledge base retrieval
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class MedicalRetriever:
    """Retrieves relevant medical information from the knowledge base"""

    def __init__(self, db_location="./medical_langchain_db", k=10):
        """
        Initialize the medical retriever

        Args:
            db_location: Path to the vector database
            k: Number of chunks to retrieve
        """
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.vector_store = Chroma(
            collection_name="medical_textbooks",
            persist_directory=db_location,
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

    def retrieve(self, question):
        """
        Retrieve relevant medical documents for a query

        Args:
            question: Patient symptom description

        Returns:
            List of relevant document chunks
        """
        return self.retriever.invoke(question)

    def format_knowledge(self, medical_docs):
        """
        Format retrieved documents for the LLM

        Args:
            medical_docs: List of document chunks

        Returns:
            Formatted string with source info and content
        """
        formatted_knowledge = ""
        for i, doc in enumerate(medical_docs, 1):
            source = doc.metadata.get('source', 'Unknown').replace('.pdf', '')
            page = doc.metadata.get('page', '?')
            formatted_knowledge += f"\n[{source} - Page {page}]\n{doc.page_content[:800]}\n{'---' * 20}\n"

        return formatted_knowledge

    def get_sources_info(self, medical_docs):
        """
        Extract source information from retrieved documents

        Args:
            medical_docs: List of document chunks

        Returns:
            Dict with source names and page numbers
        """
        sources = {}
        for doc in medical_docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')

            if source not in sources:
                sources[source] = []
            sources[source].append(page)

        return sources
