"""Exact native-model restoration extraction for LLaVA3.5-7B."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.experiment_cli import main

if __name__ == "__main__":
    main(default_model_id='llava-hf/llava-1.5-7b-hf')
