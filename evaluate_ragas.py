import os
import json
from datasets import Dataset
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever, QueryFusionRetriever
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings  # LangChain HF embeddings
from config import settings

# ================= Gemini-specific ==================
from llama_index.llms import BaseLLM  # You can create a wrapper LLM for Gemini


class GeminiLLM(BaseLLM):
    """
    Minimal Gemini wrapper for LlamaIndex
    """
    def __init__(self, api_key: str, model: str = "gemini-1.5"):
        self.api_key = api_key
        self.model_name = model

    def complete(self, prompt: str, **kwargs) -> str:
        import requests

        url = "https://api.gemini.ai/v1/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.0),
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
# ====================================================


def load_qa_dataset(dataset_path=None):
    dataset_path = dataset_path or os.path.join(settings.default_data_dir, "rag_eval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def setup_hybrid_query_engine():
    """
    Hybrid retrieval query engine:
    Dense + Sparse retrieval + optional Cohere reranker
    """
    print("🔗 Connecting to Milvus...")
    vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.collection_name,
        dim=settings.embedding_dim,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("🔍 Initializing Gemini LLM...")
    llm = GeminiLLM(api_key=os.getenv("GEMINI_API_KEY"), model="gemini-1.5")

    # Build vector index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    )

    # Dense retriever
    dense_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

    # Sparse keyword retriever
    sparse_retriever = KeywordTableSimpleRetriever(index=index, similarity_top_k=10)

    # Hybrid retriever
    hybrid_retriever = QueryFusionRetriever(retrievers=[dense_retriever, sparse_retriever])

    # Cohere reranker
    reranker = CohereRerank(api_key=settings.cohere_api_key, top_n=3)

    # Query engine
    query_engine = index.as_query_engine(
        retriever=hybrid_retriever,
        llm=llm,
        node_postprocessors=[reranker],
        response_mode="compact"
    )

    print("✅ Hybrid Query Engine Ready!")
    return query_engine


def evaluate_qa():
    print("🔍 Starting RAGAS evaluation on JSON dataset...")
    qa_pairs = load_qa_dataset()
    print(f"✅ Loaded {len(qa_pairs)} QA pairs from dataset.")

    query_engine = setup_hybrid_query_engine()

    data_dict = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for idx, pair in enumerate(qa_pairs, start=1):
        q = pair["question"]
        gt = pair["ground_truth"]
        print(f"\n[{idx}/{len(qa_pairs)}] ❓ Question: {q}")

        response = query_engine.query(q)
        rag_answer = str(response)
        contexts = [n.node.get_text() for n in response.source_nodes]

        print(f"🤖 RAG Answer: {rag_answer}")
        print(f"✅ Ground Truth: {gt}")

        data_dict["question"].append(q)
        data_dict["answer"].append(rag_answer)
        data_dict["contexts"].append(contexts)
        data_dict["ground_truth"].append(gt)

    # Prepare dataset for RAGAS
    dataset = Dataset.from_dict(data_dict)
    run_config = RunConfig(max_workers=1)
    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]

    print("\n⏳ Running RAGAS evaluation...")
    eval_result = evaluate(dataset, metrics=metrics, run_config=run_config)

    # Save and display results
    df = eval_result.to_pandas()
    output_file = "rag_evaluation_results_gemini.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Evaluation results saved to {output_file}")
    print("\n📊 Metrics per question:")
    print(df.to_string(index=False))
    print("\n📈 Overall averages:")
    print(df.mean(numeric_only=True))


if __name__ == "__main__":
    evaluate_qa()
