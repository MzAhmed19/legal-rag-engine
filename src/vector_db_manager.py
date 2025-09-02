from pymilvus import (
    connections, utility, Collection, CollectionSchema, FieldSchema, DataType
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from config import settings


class VectorDBManager:
    """Wrapper for Milvus vector store with llama_index compatibility."""

    def __init__(self):
        self.collection_name = settings.collection_name
        self.dim = settings.embedding_dim

        print(f"🔗 Connecting to Milvus at {settings.milvus_uri}...")
        connections.connect("default", uri=settings.milvus_uri)
        print(f"✅ Connected to Milvus at {settings.milvus_uri}")

        # 🔥 llama_index Milvus wrapper
        self.vector_store = MilvusVectorStore(
            uri=settings.milvus_uri,
            dim=self.dim,
            overwrite=False,
            collection_name=self.collection_name,
            index_params={"metric_type": "IP", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 256}},
        )

    def get_or_create_collection(self):
        """Ensure the Milvus collection exists."""
        if utility.has_collection(self.collection_name):
            print(f"📂 Collection '{self.collection_name}' already exists. Loading...")
            return Collection(self.collection_name)

        print(f"🆕 Creating collection '{self.collection_name}'...")
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields=fields, description="Legal document chunks for RAG")
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"✅ Collection '{self.collection_name}' created.")
        return collection
