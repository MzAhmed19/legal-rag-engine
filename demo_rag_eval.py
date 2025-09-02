import os
import time
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank

from ragas.metrics import faithfulness, context_recall, answer_relevancy
from ragas import evaluate
from ragas.run_config import RunConfig
from datasets import Dataset
from config import settings


from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.indices.base_retriever import MultiRetriever
from llama_index.retrievers import VectorIndexRetriever, KeywordTableRetriever
from config import settings

def setup_query_engine():
    """
    Hybrid RAG Query Engine: Dense (Milvus + HF embeddings) + Sparse (Keyword/BM25) + Cohere reranker
    """
    print("🔗 Connecting to Milvus...")
    vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.collection_name,
        dim=settings.embedding_dim,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("🔍 Initializing Embeddings & LLM...")
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_name)
    llm = Groq(model=settings.llm_model_name, api_key=settings.groq_api_key)

    # Build vector index
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )

    # Dense retriever
    dense_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

    # Sparse retriever (BM25-style keyword retriever)
    sparse_retriever = KeywordTableRetriever(index=index, similarity_top_k=10)

    # Fuse dense + sparse retrievers
    hybrid_retriever = MultiRetriever(retrievers=[dense_retriever, sparse_retriever])

    # Cohere reranker
    reranker = CohereRerank(api_key=settings.cohere_api_key, top_n=3)

    print("✅ Hybrid Query Engine Ready!")
    query_engine = index.as_query_engine(
        retriever=hybrid_retriever,
        llm=llm,
        node_postprocessors=[reranker],
        response_mode="compact",
    )
    return query_engine


def score_response(question, answer, contexts):
    """
    Run RAGAS evaluation without ground truth.
    """
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    dataset = Dataset.from_dict(data)
    run_config = RunConfig(max_workers=1)

    metrics = [faithfulness, context_recall, answer_relevancy]
    eval_result = evaluate(dataset, metrics=metrics, run_config=run_config)
    return eval_result.to_pandas().iloc[0].to_dict()


def run_demo():
    """
    Continuous terminal-based demo with hybrid retrieval & auto scoring.
    """
    engine = setup_query_engine()
    print("\n🤖 Legal QA Demo Ready! Type your question, or type 'exit' to quit.\n")

    while True:
        query = input("❓ Your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("\n👋 Exiting demo. Goodbye!")
            break

        if not query:
            print("⚠️ Please enter a valid question or 'exit' to quit.")
            continue

        start_time = time.time()
        print("\n🔎 Searching knowledge base...")
        response = engine.query(query)
        elapsed = time.time() - start_time

        print("\n📜 Answer:")
        print(response.response)

        contexts = [n.node.get_text() for n in response.source_nodes]

        print("\n📊 Retrieved Chunks & Scores:")
        for i, node in enumerate(response.source_nodes, start=1):
            print(f"  {i}. Score: {node.score:.4f}")
            snippet = node.node.get_text()[:200].replace("\n", " ")
            print(f"     {snippet}...")
        print(f"⏱️ Retrieval + LLM Time: {elapsed:.2f} sec")
        print("-" * 50)

        # Run evaluation
        print("\n📈 Evaluating response quality...")
        scores = score_response(query, response.response, contexts)
        print(f"   Faithfulness: {scores.get('faithfulness', 'N/A'):.3f}")
        print(f"   Context Recall: {scores.get('context_recall', 'N/A'):.3f}")
        print(f"   Answer Relevancy: {scores.get('answer_relevancy', 'N/A'):.3f}")
        print("=" * 50)
        print("\n🔁 Ask another question or type 'exit' to quit.\n")


if __name__ == "__main__":
    run_demo()
