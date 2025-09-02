import os
import re
import json
from pathlib import Path
from typing import List, Dict

MD_CLEANED_DIR = "md_cleaned"
OUTPUT_DIR = "md_structured"
CHUNK_JSON_PATH = "chunked_dataset.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_markdown_headers(text: str, act_name: str) -> str:
    """Add Markdown headers (#, ##, ###, ####) for parts, sections, and clauses."""
    lines = text.splitlines()
    formatted = [f"# {act_name}\n"]

    for line in lines:
        line = line.strip()

        # Match Part headings
        if re.match(r"^(PART|Chapter)\s+[A-Z]+", line, re.IGNORECASE):
            formatted.append(f"## {line}")

        # Match Section numbers like "5A." or "Section 5."
        elif re.match(r"^(\d+[A-Z]?\.)", line):
            section_title = re.sub(r"^(\d+[A-Z]?\.)", r"Section \1", line)
            formatted.append(f"### {section_title}")

        # Match Clause markers like "(a)" or "(1)"
        elif re.match(r"^\([a-z0-9]+\)", line, re.IGNORECASE):
            formatted.append(f"#### {line}")

        else:
            formatted.append(line)

    return "\n".join(formatted)


def chunk_markdown(content: str, source_name: str) -> List[Dict]:
    """Chunk content based on ###/#### headers, keeping metadata."""
    chunks = []
    sections = re.split(r"(?=\n### )", content)  # split by sections
    for sec in sections:
        if not sec.strip():
            continue
        clauses = re.split(r"(?=\n#### )", sec)
        for clause in clauses:
            chunk_text = clause.strip()
            if not chunk_text:
                continue

            # Build metadata
            section_match = re.search(r"### (.+)", sec)
            clause_match = re.search(r"#### (.+)", clause)

            section_title = section_match.group(1) if section_match else "Unknown Section"
            clause_title = clause_match.group(1) if clause_match else None

            meta = {
                "source": source_name,
                "section": section_title,
                "clause": clause_title,
                "content": chunk_text
            }
            chunks.append(meta)
    return chunks


def process_markdown_files():
    """Process all markdown files in md_cleaned, save structured markdown & JSON dataset."""
    all_chunks = []

    for file_name in os.listdir(MD_CLEANED_DIR):
        if not file_name.endswith(".md"):
            continue

        file_path = os.path.join(MD_CLEANED_DIR, file_name)
        act_name = Path(file_name).stem.replace("_", " ").title()

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Step 1: Add markdown headers
        structured_md = add_markdown_headers(text, act_name)

        # Save structured markdown
        out_path = os.path.join(OUTPUT_DIR, file_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(structured_md)

        # Step 2: Chunk markdown
        chunks = chunk_markdown(structured_md, act_name)
        all_chunks.extend(chunks)

    # Save JSON chunks
    with open(CHUNK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Processed all files. Structured Markdown in '{OUTPUT_DIR}', chunks in '{CHUNK_JSON_PATH}'")


if __name__ == "__main__":
    process_markdown_files()
