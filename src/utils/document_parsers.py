import os
import json
import pickle
from typing import List
from llama_index.core.schema import Document

JSON_QA_CACHE_PATH = "parsed_qa_docs.pkl"

def load_indiclegalqa_dataset(input_dir: str) -> List[Document]:
    """
    Loads the IndicLegalQA JSON dataset from a directory and caches the results.
    """
    if os.path.exists(JSON_QA_CACHE_PATH):
        print(f"✅ Loading Q&A documents from cache: {JSON_QA_CACHE_PATH}")
        with open(JSON_QA_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print(f"Cache not found. Loading IndicLegalQA dataset from: {input_dir}")
    if not os.path.isdir(input_dir):
        return []

    qa_documents = []
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json")]

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                doc = Document(
                    text=item.get('answer', ''),
                    metadata={
                        "source": "IndicLegalQA",
                        "case_name": item.get('case_name', 'Unknown Case'),
                        "related_question": item.get('question', '')
                    }
                )
                qa_documents.append(doc)
    
    print(f"Loaded {len(qa_documents)} question-context pairs from {len(json_files)} JSON file(s).")
    
    print(f"💾 Saving Q&A documents to cache: {JSON_QA_CACHE_PATH}")
    with open(JSON_QA_CACHE_PATH, "wb") as f:
        pickle.dump(qa_documents, f)

    return qa_documents