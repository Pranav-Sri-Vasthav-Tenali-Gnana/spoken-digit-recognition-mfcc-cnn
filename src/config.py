from pathlib import Path
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "fsdd_recordings"
SPEAKERS = ["jackson", "nicolas", "theo", "yweweler", "george", "lucas"]
DIGITS = list(range(10))
REPS = list(range(50))
BASE_URL = (
    "https://github.com/Jakobovski/free-spoken-digit-dataset/"
    "raw/master/recordings/"
)

SR = 8000
N_MFCC = 40
N_MELS = 64
MAX_FRAMES = 64
NUM_CLASSES = 10
N_PER_DIGIT = 15

# All figures are written here (created on first use)
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def output_path(filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename
