import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
for _extension in ('simple-knn', 'diff-plane-rasterization'):
    _path = str(_root / 'submodules' / _extension)
    if _path not in sys.path:
        sys.path.insert(0, _path)
