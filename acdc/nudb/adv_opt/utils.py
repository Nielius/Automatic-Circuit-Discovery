import os
from pathlib import Path

num_examples = 5
metric = "kl_div"
device = "cpu"

CIRCUITBENCHMARKS_DATA_DIR = Path(os.environ.get("CIRCUITBENCHMARKS_DATA_DIR", "/tmp/circuitbenchmarks_data"))
CIRCUITBENCHMARKS_DATA_DIR.mkdir(exist_ok=True)
