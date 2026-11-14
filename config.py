from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
SPLITS = DATA / "splits"
RESULTS = ROOT / "results"

VSR_URL = "https://raw.githubusercontent.com/cambridgeltl/visual-spatial-reasoning/master/data/data_files/all_vsr_validated_data.jsonl"

HELD_OUT = ["beneath", "at the right side of", "contains", "far from", "facing away from"]
VAL_FRAC = 0.1
TEST_FRAC = 0.2