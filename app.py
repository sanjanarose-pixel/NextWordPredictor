"""A small Streamlit interface for the locally trained language model."""

from html import escape
import re

import streamlit as st

from v3_inference import generate_text, load_v3_model


def clean_continuation(text):
    """Remove WikiText heading markers and stray punctuation at the start."""
    # WikiText headings use standalone runs of equals signs as delimiters.
    text = re.sub(r"(?<!\S)=+(?=\s|$)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Keep punctuation inside the sentence; trim only an awkward start.
    leading_punctuation = ".,!?:;%)]}"
    while text and text[0] in leading_punctuation:
        text = text[1:].lstrip()
    return text


st.set_page_config(page_title="Next Word Predictor", layout="centered")
st.markdown(
    """
    <style>
    .block-container { max-width: 760px; padding-top: 3rem; padding-bottom: 3rem; }
    h1 { letter-spacing: -0.03em; }
    .subtitle { color: #667085; font-size: 1.05rem; margin-top: -0.5rem; }
    .result-card {
        background: #f4f7fb;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        line-height: 1.7;
        min-height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Next Word Predictor")
st.markdown(
    '<p class="subtitle">Generate text using my locally trained Transformer language model.</p>',
    unsafe_allow_html=True,
)


@st.cache_resource
def get_trained_model():
    """Load the Version 3 paragraph-trained checkpoint; startup never trains."""
    return load_v3_model()


try:
    model, tokenizer, _ = get_trained_model()
except Exception as error:
    st.error(
        "The trained model could not be loaded. Run "
        f"train_v3_model.py first. Details: {error}"
    )
    st.stop()


prompt = st.text_area(
    "Enter your prompt",
    placeholder="Start typing here...",
    height=140,
)
context_window = st.slider(
    "Context window",
    min_value=1,
    max_value=64,
    value=32,
    help="How many recent tokens the model considers for each next token.",
)
number_of_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=100,
    value=10,
)

generate_clicked = st.button("Generate", type="primary", use_container_width=True)

if generate_clicked:
    if not prompt.strip():
        st.error("Please enter a prompt before generating text.")
    else:
        try:
            continuation = generate_text(
                model,
                tokenizer,
                prompt,
                number_of_words=number_of_words,
                context_window=context_window,
            )
            continuation = clean_continuation(continuation)
            st.subheader("Generated continuation")
            st.markdown(
                f'<div class="result-card">{escape(continuation)}</div>',
                unsafe_allow_html=True,
            )
        except (ValueError, RuntimeError) as error:
            st.error(f"Text generation failed: {error}")
