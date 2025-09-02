import os
import json
from tqdm import tqdm

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import settings


def load_chunked_json_files(folder_path):
    """
    Load all chunked JSON files and convert them into LlamaIndex Document objects.
    """
    docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith("_fixed.json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for entry in data:
                clause = entry.get("clause", "Untitled Section")
                content = entry.get("content", "No content available")
                chunk_id = entry.get("chunk_id", "no_id")

                # Convert to LlamaIndex Document
                docs.append(
                    Document(
                        text=content,
                        metadata={
                            "source_file": filename,
                            "clause": clause,
                            "chunk_id": chunk_id,
                        },
                    )
                )
    return docs


def create_milvus_index(docs):
    """
    Create Milvus index and store embeddings.
    """
    print("🔗 Connecting to Milvus & initializing embedding model...")

    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_name)

    vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.collection_name,
        dim=settings.embedding_dim,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("📥 Ingesting documents into Milvus...")
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print("✅ Successfully ingested all documents into Milvus!")
    return index


def run_ingestion():
    """
    Main function to run JSON ingestion pipeline.
    """
    INPUT_FOLDER = "src"  # Folder containing chunked JSON
    print(f"📂 Loading chunked JSON files from: {INPUT_FOLDER}")
    docs = load_chunked_json_files(INPUT_FOLDER)

    print(f"📄 Loaded {len(docs)} document chunks. Starting ingestion...")
    create_milvus_index(docs)
    print("🎉 Ingestion completed!")


if __name__ == "__main__":
    run_ingestion()
