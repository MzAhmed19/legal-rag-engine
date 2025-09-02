import typer
from typing_extensions import Annotated
from config import settings
from src.document_processor import NodeProcessor, load_markdown_documents
from src.vector_db_manager import VectorDBManager
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as LlamaIndexGroq
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine

app = typer.Typer(help="📚 Legal Document RAG Pipeline CLI")

@app.command(help="Ingest cleaned Markdown files into Milvus Vector DB")
def ingest(
    md_dir: Annotated[str, typer.Option("--md-dir", help="Directory with cleaned Markdown files")] = "src/md_cleaned/",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
):
    print("--- Starting Data Ingestion from Cleaned Files ---")

    md_documents = load_markdown_documents(md_dir)
    if not md_documents:
        print("❌ No documents found. Exiting.")
        raise typer.Exit()

    processor = NodeProcessor(settings.embedding_model_name, verbose=verbose)
    safe_nodes = processor.process_documents(md_documents)
    nodes_for_milvus = processor.prepare_for_milvus(safe_nodes)

    print(f"📊 Ready {len(nodes_for_milvus)} nodes for Milvus insertion.")

    vdb = VectorDBManager()
    vdb.get_or_create_collection()

    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_name)
    storage_context = StorageContext.from_defaults(vector_store=vdb.vector_store)

    VectorStoreIndex(
        nodes_for_milvus,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    print("\n✅ Ingestion Finished! Documents are chunked, embedded, and stored in Milvus.")
    from pymilvus import connections, Collection

    connections.connect(uri=settings.milvus_uri)
    collection = Collection(settings.collection_name)
    print(f"📦 Total documents in Milvus: {collection.num_entities}")


@app.command(help="Query the Milvus Vector DB for relevant answers")
def query(
    question: Annotated[str, typer.Argument(help="Your query/question")],
    top_k: Annotated[int, typer.Option("--top-k", help="Number of results to retrieve")] = 3
):
    print(f"🔍 Querying Milvus for: {question}")

    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_name)
    vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.collection_name,
        dim=settings.embedding_dim,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    llm = LlamaIndexGroq(model=settings.llm_model_name, api_key=settings.groq_api_key)
    reranker = CohereRerank(api_key=settings.cohere_api_key, top_n=3)
    window_postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")

    query_engine = RetrieverQueryEngine.from_args(
        retriever=index.as_retriever(similarity_top_k=top_k),
        node_postprocessors=[window_postprocessor, reranker],
        llm=llm,
    )

    response = query_engine.query(question)
    print("\n--- 📜 Top Results ---")
    print(response)

if __name__ == "__main__":
    app()  # <-- this must be here
