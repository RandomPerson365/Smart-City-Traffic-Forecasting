from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT/'data'
RAW_DATA = DATA_DIR/'raw'
PROCESSED_DATA = DATA_DIR/'processed'
MODEL_DIR = ROOT/'models'
OUTPUT_DIR = ROOT/'outputs'
FIGURE_DIR = OUTPUT_DIR/'figures'
REPORT_DIR = OUTPUT_DIR/'reports'
SUBMISSION_DIR = OUTPUT_DIR/'submissions'
RANDOM_STATE = 42
LOOKBACK = 24
N_SPLITS = 5
