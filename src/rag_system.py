"""
RAG (Retrieval Augmented Generation) System
Medical knowledge retrieval using ChromaDB and sentence transformers.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os


class MedicalRAG:
    """RAG system for medical document retrieval."""

    def __init__(self, collection_name="medical_knowledge", persist_directory="./chroma_db"):
        """
        Initialize the RAG system with ChromaDB.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Where to store the vector database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize sentence transformer for embeddings
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection '{collection_name}' with {self.collection.count()} documents")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Medical knowledge base for RAG"}
            )
            print(f"Created new collection '{collection_name}'")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add medical documents to the vector database.

        Args:
            documents: List of document dictionaries with 'id', 'title', 'content', 'category'
        """
        if not documents:
            print("No documents to add")
            return

        print(f"Adding {len(documents)} documents to the knowledge base...")

        ids = []
        contents = []
        metadatas = []

        for doc in documents:
            ids.append(doc['id'])
            contents.append(doc['content'])
            metadatas.append({
                'title': doc['title'],
                'category': doc.get('category', 'general')
            })

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(f"Successfully added {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant documents for a query.

        Args:
            query: User's medical question
            top_k: Number of top documents to retrieve

        Returns:
            List of relevant documents with metadata and similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Format results
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'title': results['metadatas'][0][i].get('title', 'Unknown'),
                    'category': results['metadatas'][0][i].get('category', 'general'),
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })

        return retrieved_docs

    def get_context_string(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve documents and format as a context string for LLM.

        Args:
            query: User's medical question
            top_k: Number of documents to retrieve

        Returns:
            Formatted context string
        """
        docs = self.retrieve(query, top_k=top_k)

        if not docs:
            return "No relevant medical information found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}: {doc['title']}]\n{doc['content']}\n")

        return "\n".join(context_parts)

    def clear_collection(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Medical knowledge base for RAG"}
        )
        print(f"Cleared collection '{self.collection_name}'")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "embedding_model": "all-MiniLM-L6-v2",
            "persist_directory": self.persist_directory
        }


def initialize_medical_rag(force_reload=False):
    """
    Initialize the Medical RAG system and load documents if needed.

    Args:
        force_reload: If True, clear existing data and reload all documents

    Returns:
        MedicalRAG instance
    """
    from data.medical_knowledge import get_all_documents

    rag = MedicalRAG()

    # Check if we need to load documents
    if force_reload or rag.collection.count() == 0:
        if force_reload:
            print("Force reload: clearing existing data...")
            rag.clear_collection()

        print("Loading medical documents...")
        documents = get_all_documents()
        rag.add_documents(documents)

    print("\nRAG System Stats:")
    print(rag.get_stats())

    return rag


if __name__ == "__main__":
    # Test the RAG system
    print("=== Medical RAG System Test ===\n")

    # Initialize
    rag = initialize_medical_rag(force_reload=True)

    # Test query
    test_query = "What are the symptoms of depression?"
    print(f"\nQuery: {test_query}")
    print("\nRetrieved Documents:")

    results = rag.retrieve(test_query, top_k=2)
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['title']} (Category: {doc['category']})")
        print(f"   Distance: {doc['distance']:.4f}")
        print(f"   Preview: {doc['content'][:200]}...")

    # Test context string
    print("\n" + "=" * 70)
    print("Context String for LLM:")
    print("=" * 70)
    context = rag.get_context_string(test_query, top_k=1)
    print(context)
