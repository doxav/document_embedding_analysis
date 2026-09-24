from pathlib import Path
import sys
root=str(Path(__file__).resolve().parents[1])
if root not in sys.path:sys.path.insert(0,root)
from posttrain.evaluation import get_assert
