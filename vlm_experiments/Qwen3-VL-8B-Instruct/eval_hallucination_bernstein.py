"""Model-independent empirical Bernstein evaluation of saved patch effects."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.bernstein import main, run_bernstein_evaluation

if __name__ == "__main__":
    main()
