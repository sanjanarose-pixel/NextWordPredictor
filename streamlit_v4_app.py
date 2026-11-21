"""Standalone Streamlit demo for the TinyStories V4 model."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from v4_inference import generate, load_v4


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "artifacts" / "v4" / "v4_best_20261121_170608_22072_1b90f1c4.pth"
TOKENIZER = ROOT / "artifacts" / "v4" / "tokenizer_v4_20261121_170608_22072_1b90f1c4.json"
EXAMPLE_PROMPTS = [
    "Once upon a time, there was a little girl who",
    "One morning, a rabbit found a strange key",
    "Deep inside the forest, there lived",
    "The little dog followed the sound",
    "A little boy went into the forest and",
    "One day, a small robot decided to",
]


@st.cache_resource(show_spinner="Loading the V4 story model…")
def get_model():
    return load_v4(CHECKPOINT, TOKENIZER, device="cpu")


st.set_page_config(page_title="TinyStories V4", page_icon="📚", layout="centered")
st.title("TinyStories V4")
st.caption("A small children's-story language model trained from random initialization.")
st.markdown(
    "The model learned from a 30,000-story subset of synthetic GPT-4-generated "
    "TinyStories. It is a compact demonstration, so continuations can be odd or "
    "grammatically imperfect. No prompt is sent to an external service."
)

selected = st.selectbox("Try an example prompt", ["Choose an example…", *EXAMPLE_PROMPTS])
default_prompt = "" if selected == "Choose an example…" else selected
prompt = st.text_area("Your story opening", value=default_prompt, height=100,
                      placeholder="Once upon a time, there was a little rabbit who")

if st.button("Continue story", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Enter a story opening first.")
    else:
        try:
            model, tokenizer, _ = get_model()
            with st.spinner("Writing a short continuation…"):
                full_text = generate(model, tokenizer, prompt, max_new_tokens=48,
                                     temperature=0.8, top_k=40, seed=42,
                                     min_new_words=5, max_new_words=9)
            continuation = full_text[len(prompt):]
            st.subheader("Continuation")
            st.write(continuation if continuation.strip() else "The model returned an empty continuation.")
        except FileNotFoundError as exc:
            st.error(f"The model files are missing: {exc}")

st.divider()
st.caption(
    "Dataset: [TinyStoriesV2_cleaned](https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned) "
    "by Fachhochschule Südwestfalen, licensed under [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/). "
    "The source stories are not included in this repository."
)
