# V4 TinyStories children's-story model

See the [project README](README.md) for the full setup, V1–V4 overview, dataset attribution and license notes, measured training results, sample continuations, tests, and Streamlit Community Cloud steps.

## Quick start

```powershell
python -m pip install -r requirements.txt
python demo_v4.py --prompt "Once upon a time, there was a little rabbit who"
python -m streamlit run streamlit_v4_app.py
```

The demo loads the committed checkpoint and matching tokenizer from `artifacts/v4/`. It does not train or download data. The V4 Streamlit app is separate from the existing V3 app in `app.py`.

## Model and data

V4 is a randomly initialized, CPU-trained decoder-only Transformer with 237,312 parameters: a 2,048-ID byte-fallback BPE tokenizer, dimension 64, two blocks, four heads, context length 64, dropout 0.1, and tied token embeddings. It trained for four epochs and 19,452 steps on 25,000 train stories from a deterministic 30,000-story subset of `fhswf/TinyStoriesV2_cleaned`; 2,500 stories each are held out for validation and test. Actual token counts are 9,051,969 / 915,071 / 907,851. Losses are 1.9901 / 1.8338 / 1.8420 (train / best validation / test). The combined CPU training time was 53 minutes 23 seconds.

The stories are synthetic GPT-4-generated content. Attribution: *TinyStoriesV2_cleaned*, Fachhochschule Südwestfalen (fhswf), whose publisher acknowledges the original TinyStories authors. License: [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/); source: [Hugging Face dataset card](https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned). Raw source stories and prepared story documents are not included. If redistributing source or modified dataset material, follow the license's attribution, modification-marking, and sharing terms. See the root README for details.

The 15 fixed-prompt evaluation had no empty outputs, repeated trigrams, or repeated-punctuation cases; one output had unbalanced quotation marks. Human inspection still found relevance, grammar, and continuity errors. This is a small-domain demonstrator, not a general-purpose model. Actual outputs and checks are in `artifacts/v4/v4_tinystories_eval_20261121.json`.

## Recreate training

Dataset acquisition and training are not required to run inference. If recreating them, review the source license first:

```powershell
python download_v4_tinystories.py
python prepare_v4_tinystories.py
python train_v4_model.py
python evaluate_v4.py
```
