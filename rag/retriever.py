import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(PROJECT_ROOT, "vector_store", "simple_property_chroma")

CHROMA_COLLECTION = "property_listings"

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings,
    )
