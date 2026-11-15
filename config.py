from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
SPLITS = DATA / "splits"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
VSR_URL = "https://raw.githubusercontent.com/cambridgeltl/visual-spatial-reasoning/master/data/data_files/all_vsr_validated_data.jsonl"

PROMPT = 'Statement: "{caption}"\nIs the statement true for this image? Answer with one word, yes or no.'
YES = "yes"
NO = "no"

MAX_SIDE = 448

HELD_OUT = ["beneath", "at the right side of", "contains", "far from", "facing away from"]
VAL_FRAC = 0.1
TEST_FRAC = 0.2

SEED = 0
RANK = 16
ALPHA_PER_RANK = 2
LORA_DROPOUT = 0.05
TARGETS = ["q_proj", "v_proj"]

EPOCHS = 3
BATCH = 2
ACCUM = 8
LR = 1e-4
WARMUP = 0.06
WEIGHT_DECAY = 0.0
CLIP = 1.0
LOG_EVERY = 10
WORKERS = 4

BINS = 15