import pandas as pd
import shutil
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .retriever import CHROMA_COLLECTION, PROJECT_ROOT, VECTOR_DIR

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "simple_house_properties_clean.csv"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_property_rows():
    rows = pd.read_csv(DATA_PATH)
    rows = rows.dropna(subset=["city", "price_lakh", "area_sqft", "bhk"])
    return rows

def create_documents(rows):
    documents = []
    for index, row in enumerate(rows.itertuples(index=False), start=0):
        city = row.city
        bhk = float(row.bhk)
        area_sqft = float(row.area_sqft)
        price_lakh = float(row.price_lakh)
        furnishing = row.furnishing
        description = row.description

        text = (
            f"Property in {city}: {bhk:g} BHK, "
            f"area {area_sqft:.0f} sqft, "
            f"price INR {price_lakh:.2f} lakh, "
            f"furnishing {furnishing}. "
            f"Description: {description}"
        )

        metadata = {
            "source_id": f"property_{index}",
            "city": city,
            "bhk": bhk,
            "price_lakh": price_lakh,
        }

        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def rebuild_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )

    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)

    return Chroma.from_documents(
        documents=documents,
        ids=[document.metadata["source_id"] for document in documents],
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(VECTOR_DIR),
    )

def main():
    rows = load_property_rows()
    documents = create_documents(rows)
    rebuild_vector_store(documents)

    print(f"Documents: {len(documents):,}")
    print(f"ChromaDB: {VECTOR_DIR}")

if __name__ == "__main__":
    main()
