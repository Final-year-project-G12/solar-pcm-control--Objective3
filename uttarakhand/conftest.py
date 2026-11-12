"""pytest: put the Uttarakhand O3 root on sys.path (o3.o2_bridge adds the frozen O2 snapshot itself)."""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
