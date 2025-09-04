import os
import time
from datasets import Dataset
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from llama_index.postprocessor.cohere_rerank import CohereRerank
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from ragas.run_config import RunConfig
from config import settings

# 🔹 Define a strict system prompt to keep answers grounded
SYSTEM_PROMPT = """
You are a legal assistant AI.
Your goal is to answer questions based on the retrieved context provided below.
Answer ONLY based on the retrieved context provided below.
Do NOT add extra information or assumptions.
You can add only if it is right and relevant. 
If the context doesnt make sense with the question, provide the answer with the right context only.
Be concise, structured, and factual but make sure the legal terms and sections are simplified to be easily understood.
"""

# 🔹 Setup query engine
def setup_query_engine():
    print("🔗 Connecting to Milvus...")
    vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.collection_name,
        dim=settings.embedding_dim,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("🔍 Initializing Embeddings & LLM...")
    embed_model = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )

    reranker = CohereRerank(api_key=settings.cohere_api_key, top_n=5)

    groq_llm = ChatGroq(model=settings.llm_model_name, api_key=settings.groq_api_key)

    query_engine = index.as_query_engine(
        llm=groq_llm,
        similarity_top_k=15,
        node_postprocessors=[reranker],
        response_mode="compact",
    )

    print("✅ Query Engine Ready!")
    return query_engine, groq_llm

# 🔹 Force LLM to follow system prompt
def get_prompted_response(question, source_nodes, llm):
    context_text = "\n\n".join([n.node.get_text() for n in source_nodes])
    full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    return llm.invoke(full_prompt).content

# 🔹 Evaluate response
def score_response(question, answer, contexts):
    ragas_llm = ChatGroq(model=settings.llm_model_name, api_key=settings.groq_api_key)
    ragas_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    run_config = RunConfig(max_workers=1)

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy]
    eval_result = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )
    return eval_result.to_pandas().iloc[0].to_dict()

# 🔹 Demo runner
def run_demo():
    engine, llm = setup_query_engine()
    print("\n🤖 Legal QA Demo Ready! Type your question, or 'exit' to quit.\n")

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
        raw_response = engine.query(query)
        elapsed = time.time() - start_time

        answer = get_prompted_response(query, raw_response.source_nodes, llm)
        contexts = [n.node.get_text() for n in raw_response.source_nodes]

        print("\n📜 Answer:")
        print(answer)

        print("\n📊 Retrieved Chunks:")
        for i, node in enumerate(raw_response.source_nodes, start=1):
            print(f"  {i}. Score: {node.score:.4f}")
            snippet = node.node.get_text()[:200].replace("\n", " ")
            print(f"     {snippet}...")
        print(f"⏱️ Retrieval + LLM Time: {elapsed:.2f} sec")
        print("-" * 50)

        # RAGAS scoring
        print("\n📈 Evaluating response quality...")
        scores = score_response(query, answer, contexts)
        print(f"   Faithfulness: {scores.get('faithfulness', 'N/A'):.3f}")
        print(f"   Answer Relevancy: {scores.get('answer_relevancy', 'N/A'):.3f}")
        print("=" * 50)
        print("\n🔁 Ask another question or type 'exit' to quit.\n")

if __name__ == "__main__":
    run_demo()
