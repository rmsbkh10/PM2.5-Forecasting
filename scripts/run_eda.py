"""Part A: run the full exploration and preprocessing pipeline.

Usage (from the project root):
    python scripts/run_eda.py
The dataset must be in data/raw/combined_gee_pm25.csv.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.part_a import run_part_a  # noqa: E402

if __name__ == "__main__":
    result = run_part_a()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str))
    print("\nPart A finished: see results/G22/, folds/, figures/ and data/processed/.")
