"""Load Version 3 artifacts while retaining the exact Version 2 decoder."""

from high_quality_inference import generate_text
from high_quality_inference import load_high_quality_model


CHECKPOINT_FILE = "v3_model.pth"
TOKENIZER_FILE = "v3_tokenizer.json"


def load_v3_model():
    """Load the separate Version 3 model and its matching tokenizer."""
    return load_high_quality_model(CHECKPOINT_FILE, TOKENIZER_FILE)
