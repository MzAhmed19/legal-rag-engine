import json
import pandas as pd
from datasets import Dataset

# 🔹 LlamaIndex core + Milvus
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

# 🔹 LangChain HuggingFace Embeddings (correct one)
from langchain_huggingface import HuggingFaceEmbeddings

# 🔹 LLMs
from llama_index.llms.groq import Groq
from langchain_groq import ChatGroq

# 🔹 RAGAS evaluation
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
)
from ragas.run_config import RunConfig

# 🔹 Cohere Reranker
from llama_index.postprocessor.cohere_rerank import CohereRerank

# 🔹 Local settings
from config import settings



# ✅ Load QA Dataset
def load_qa_data(dataset_path=None):
    dataset_path = dataset_path or f"{settings.default_data_dir}/dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ✅ Create Query Engine with Reranker
def create_query_engine():
    # Embedding model
    embed_model = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    # Milvus vector store
    vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.collection_name,
        dim=settings.embedding_dim,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build vector index
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )

    # LLM
    groq_llm = Groq(model=settings.llm_model_name, api_key=settings.groq_api_key)

    # Cohere reranker
    reranker = CohereRerank(api_key=settings.cohere_api_key, top_n=5)

    # Query engine with retrieval + reranking
    return index.as_query_engine(
        llm=groq_llm,
        similarity_top_k=12,  # Retrieve top 10 candidates
        node_postprocessors=[reranker],  # Re-rank & return top 3
    )


# ✅ Prepare Dataset for RAGAS
def prepare_dataset(query_engine, qa_pairs):
    data_dict = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for idx, pair in enumerate(qa_pairs, start=1):
        q = pair["question"]
        gt = pair["ground_truth"]

        response = query_engine.query(q)
        rag_answer = str(response)
        contexts = [n.node.get_content() for n in response.source_nodes]

        print(f"\n[{idx}/{len(qa_pairs)}] ❓ Q: {q}")
        print(f"🤖 RAG Answer: {rag_answer}")
        print(f"✅ Ground Truth: {gt}")

        data_dict["question"].append(q)
        data_dict["answer"].append(rag_answer)
        data_dict["contexts"].append(contexts)
        data_dict["ground_truth"].append(gt)

    return Dataset.from_dict(data_dict)


# ✅ Run Evaluation
def run_evaluation():
    print("🔍 Starting Legal QA Evaluation with RAGAs metrics...")

    qa_pairs = load_qa_data()
    print(f"✅ Loaded {len(qa_pairs)} QA pairs from rag_eval_dataset.json")

    query_engine = create_query_engine()
    print("✅ Query engine with Groq LLM + HF embeddings + Cohere reranker ready!")

    ds = prepare_dataset(query_engine, qa_pairs)

    # 👇 Explicitly force RAGAS to use Groq + HF
    ragas_llm = ChatGroq(model=settings.llm_model_name, api_key=settings.groq_api_key)
    ragas_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    run_config = RunConfig(max_workers=1)

    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]
    print("\n⏳ Running RAGAs evaluation...")
    eval_result = ragas_evaluate(
        ds,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    # Convert results to DataFrame
    df = eval_result.to_pandas()
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
