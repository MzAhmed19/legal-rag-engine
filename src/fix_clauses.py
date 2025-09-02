import json
import re
from pathlib import Path

def fix_clauses(json_path: str, output_path: str = None):
    """
    Fixes null clause fields in a JSON dataset by inferring clause numbers
    or assigning default values.
    """

    # Load data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Regex patterns for clause detection
    clause_patterns = [
        r'^\d+(\.\d+)*',          # e.g., 1, 1.1, 2.3.4
        r'^[A-Z]\d*',             # e.g., A, B1
        r'^[IVXLC]+\.',           # Roman numerals I., II., III.
        r'^Section\s+\d+',        # Section 1, Section 45
    ]

    def detect_clause(text):
        """Detect clause number from text using regex patterns."""
        for pattern in clause_patterns:
            match = re.match(pattern, text.strip())
            if match:
                return match.group()
        return None

    last_clause = "Unlabeled Section"
    for entry in data:
        # If clause is null or empty, try to detect it
        if not entry.get("clause"):
            content = entry.get("content", "")
            detected = detect_clause(content)
            if detected:
                entry["clause"] = detected
                last_clause = detected
            else:
                entry["clause"] = last_clause  # fallback to previous clause

    # Save fixed file
    output_path = output_path or json_path.replace(".json", "_fixed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Fixed clauses saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    input_file = "chunked_dataset.json"  # Change to your actual file path
    fix_clauses(input_file, "data_fixed.json")
