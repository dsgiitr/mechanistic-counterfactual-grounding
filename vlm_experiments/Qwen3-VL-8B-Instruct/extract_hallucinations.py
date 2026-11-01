import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.hallucination_cli import main

if __name__ == "__main__":
    main("Qwen/Qwen3-VL-8B-Instruct")
