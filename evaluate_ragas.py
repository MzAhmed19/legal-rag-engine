import json
import pandas as pd
from datasets import Dataset

# 🔹 LlamaIndex & Milvus
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleKeywordTableIndex,
    QueryBundle,
)
from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.retrievers import QueryFusionRetriever

# 🔹 RAGAS
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from ragas.run_config import RunConfig

# 🔹 LangChain (for evaluation embeddings/LLM)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# 🔹 Project Config
from config import settings


# ✅ Load QA Dataset
def load_qa_data(dataset_path=None):
    """
    Load evaluation questions and ground truths from rag_eval_dataset.json
    """
    dataset_path = dataset_path or f"{settings.default_data_dir}/rag_eval_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ✅ Create Hybrid Query Engine
def create_query_engine():
    """
    Hybrid Retrieval: Combines Vector Search (Milvus) + Keyword Search
    with QueryFusionRetriever and Cohere Reranker.
    """
    print("🔧 Initializing Hybrid Query Engine...")

    # Embedding model
    embed_model = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    # Milvus vector store
    vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.collection_name,
        dim=settings.embedding_dim,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Vector Index
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )

    # Keyword Index
    keyword_index = SimpleKeywordTableIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )

    # Fusion Retriever
    vector_retriever = vector_index.as_retriever(similarity_top_k=10)
    keyword_retriever = keyword_index.as_retriever(similarity_top_k=10)

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        similarity_top_k=12,
        num_queries=3,
        mode="relative_score",
    )

    # Groq LLM
    groq_llm = Groq(model=settings.llm_model_name, api_key=settings.groq_api_key)

    # Cohere Reranker
    reranker = CohereRerank(api_key=settings.cohere_api_key, top_n=5)

    print("✅ Hybrid Query Engine Ready: Vector + Keyword + Reranker")
    return hybrid_retriever, groq_llm, reranker


# ✅ Prepare Dataset for RAGAS
def prepare_dataset(hybrid_retriever, llm, reranker, qa_pairs):
    """
    Query the Hybrid RAG system for each QA pair and store results for RAGAS.
    """
    data_dict = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for idx, pair in enumerate(qa_pairs, start=1):
        q = pair["question"]
        gt = pair["ground_truth"]

        try:
            query_bundle = QueryBundle(q)
            retrieved_nodes = hybrid_retriever.retrieve(query_bundle)
            reranked_nodes = reranker.postprocess_nodes(retrieved_nodes)

            context_chunks = [n.node.get_content() for n in reranked_nodes]

            # Generate LLM answer
            answer_prompt = (
                f"Answer the following legal question based only on the provided context.\n\n"
                f"Question: {q}\n\nContext:\n"
                + "\n".join(context_chunks)
            )
            response = llm.complete(answer_prompt)
            rag_answer = response.text.strip()

            print(f"\n[{idx}/{len(qa_pairs)}] ❓ Q: {q}")
            print(f"🤖 RAG Answer: {rag_answer}")
            print(f"✅ Ground Truth: {gt}")

            data_dict["question"].append(q)
            data_dict["answer"].append(rag_answer)
            data_dict["contexts"].append(context_chunks)
            data_dict["ground_truth"].append(gt)

        except Exception as e:
            print(f"❌ Error querying for Q{idx}: {e}")
            data_dict["question"].append(q)
            data_dict["answer"].append("ERROR")
            data_dict["contexts"].append([])
            data_dict["ground_truth"].append(gt)

    return Dataset.from_dict(data_dict)


# ✅ Run Evaluation
def run_evaluation():
    """
    Full Hybrid Retrieval RAG Evaluation Pipeline
    """
    print("🔍 Starting Legal QA Evaluation with RAGAS metrics...")

    qa_pairs = load_qa_data()
    print(f"✅ Loaded {len(qa_pairs)} QA pairs from rag_eval_dataset.json")

    hybrid_retriever, groq_llm, reranker = create_query_engine()

    ds = prepare_dataset(hybrid_retriever, groq_llm, reranker, qa_pairs)

    # Use Groq LLM & HF embeddings inside RAGAS evaluation
    ragas_llm = ChatGroq(model=settings.llm_model_name, api_key=settings.groq_api_key)
    ragas_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    run_config = RunConfig(max_workers=1)

    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]

    print("\n⏳ Running RAGAS evaluation...")
    eval_result = ragas_evaluate(
        ds,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    # Convert results to DataFrame
    df = eval_result.to_pandas()

    # Clean up columns
    expected_cols = ["question", "faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    available_cols = [c for c in expected_cols if c in df.columns]
    df_clean = df[available_cols]

    # Save results
    output_file = "rag_evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Full evaluation results saved to {output_file}")

    # Show metrics
    print("\n📊 Per-Question Metrics:")
    print(df_clean.to_string(index=False))

    print("\n📈 Overall Averages:")
    print(df_clean.mean(numeric_only=True))


if __name__ == "__main__":
    run_evaluation()
