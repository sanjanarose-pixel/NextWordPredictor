# NextWordPredictor

NextWordPredictor is a collection of small language-model experiments built with PyTorch. The current focus is **V4**, a from-scratch, CPU-trained children's-story continuation model. The repository also retains the earlier V1–V3 training tracks and the existing V3 Streamlit app.

V4 is intentionally modest: it generates short continuations, not complete stories. Its outputs can be irrelevant, ungrammatical, or inconsistent. It is a learning project and is not a general-purpose language model.

## Versions

| Version | Approach and data | Current role |
|---|---|---|
| V1 | Word-level next-token model with a vocabulary built from the original text corpus. | Original experiment and checkpoint retained. |
| V2 | Sentence-level/BPE experiment (`high_quality_*`); the comparison script uses this model under the label V2. | Separate experiment and checkpoint retained. |
| V3 | 8,000-symbol BPE model trained on filtered FineWeb-Edu paragraphs. | Existing `app.py` remains the V3 app and uses `v3_model.pth` with `v3_tokenizer.json`. |
| V4 | 2,048-symbol byte-fallback BPE model trained from random initialization on a deterministic TinyStories subset. | New short children's-story demo in `demo_v4.py` and `streamlit_v4_app.py`. |

V4 does not replace the V3 app or load V1–V3 weights. The two Streamlit entrypoints are separate.

## V4 dataset, attribution, and license

V4 uses 30,000 cleaned synthetic children's stories from [`fhswf/TinyStoriesV2_cleaned`](https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned). The dataset publisher describes the stories as generated with GPT-4 and acknowledges the original TinyStories authors. Dataset license: **CDLA-Sharing-1.0**. The dataset card and [full license text](https://cdla.dev/sharing-1-0/) are linked here for attribution and terms.

Attribution: *TinyStoriesV2_cleaned*, Fachhochschule Südwestfalen (fhswf), synthetic GPT-4-generated stories; the publisher acknowledges the original TinyStories authors. Source: [dataset card](https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned).

The local subset was selected deterministically from one official train shard and one official test shard. It contains 25,000 train stories, 2,500 validation stories, and 2,500 test stories. Exact source checksums and preparation counts are in [`data/v4/tinystories_manifest.json`](data/v4/tinystories_manifest.json). The raw source downloads and prepared story text are deliberately not included in this repository. The tokenizer was fitted on train stories only; stories and token windows remain document-independent.

CDLA-Sharing-1.0 permits use under its stated conditions. If source or modified dataset material is redistributed, the license requires preserving attribution and source links, marking modifications, and sharing the material under the unmodified license. Its Results clause addresses computational results that contain no more than a de minimis portion of source data. This repository does not redistribute the story files. The model checkpoint is provided as a trained artifact; the project does not make a legal determination about every possible downstream use. Review the [license](https://cdla.dev/sharing-1-0/) and [dataset card](https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned) before redistributing dataset material.

## V4 model and measured results

- Decoder-only causal Transformer, randomly initialized and trained locally with PyTorch on CPU.
- 2,048-token byte-fallback BPE vocabulary; tokenizer and model checkpoint are a matched pair.
- Embedding dimension 64, two Transformer blocks, four attention heads, context length 64, dropout 0.1, tied input/output token embeddings.
- **237,312 parameters**, counted from the implementation.
- Train/validation/test: 25,000 / 2,500 / 2,500 stories and 9,051,969 / 915,071 / 907,851 actual tokenizer IDs, respectively (including EOS).
- Seed 2024, four CPU threads, batch size 32, AdamW learning rate 0.0005, four epochs and 19,452 completed steps. One bounded continuation epoch resumed the best model checkpoint with a new AdamW optimizer; all weights remained locally trained from random initialization.
- Final train loss 1.9901; best validation loss 1.8338; held-out test loss 1.8420. Combined training wall time was 53 minutes 23 seconds.

Loss does not establish story quality. A fresh-process evaluation used 15 fixed prompts and generated 8–9 words each. The saved evaluation has zero empty outputs, zero repeated trigrams, zero repeated-punctuation cases, and one unbalanced quotation. Human review found some plausible short story openings as well as agreement errors, odd transitions, and weak relevance. Familiar animal, forest, and everyday prompts tended to work better than specific robot/fantasy prompts and dialogue. See [`artifacts/v4/v4_tinystories_eval_20261121.json`](artifacts/v4/v4_tinystories_eval_20261121.json) for all actual outputs and checks and [`artifacts/v4/v4_results_20261121_170608_22072_1b90f1c4.json`](artifacts/v4/v4_results_20261121_170608_22072_1b90f1c4.json) for losses and counts.

### Actual sample continuations

These are samples from the saved checkpoint; they are not edited examples.

| Prompt | Generated continuation |
|---|---|
| Once upon a time, there was a little girl who | loved to play outside. She liked to read her |
| One morning, a rabbit found a strange key | in the sky. It was very pretty, but it |
| Deep inside the forest, there lived | a little boy named Tim. Tim loved to share |
| One day, a small robot decided to | go to the park. The car would sit on |
| The princess opened the mysterious box and | saw a big tree. The busy was very sad. |
| A little bird fell from its nest | and said, "Mia! I am not bored, but it |

## Install and run locally

Python 3.12 is a suitable environment for local setup and matches Streamlit Community Cloud's current default. On Windows PowerShell, run from the repository folder:

```powershell
git clone https://github.com/sanjanarose-pixel/NextWordPredictor.git
cd NextWordPredictor
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the one-prompt V4 demo:

```powershell
python demo_v4.py --prompt "Once upon a time, there was a little rabbit who"
```

Omit `--prompt` for the interactive prompt loop. To run the separate V4 web demo locally:

```powershell
python -m streamlit run streamlit_v4_app.py
```

The Streamlit demo uses only the saved V4 checkpoint and tokenizer. It does not download training data, train at startup, or call an external API. The existing V3 interface remains available with `python -m streamlit run app.py`; it requires the V3 checkpoint and tokenizer at the repository root.

## Training and evaluation commands

Recreating V4 requires obtaining the dataset files separately and accepting their license. The downloader fetches only the two selected official source shards; preparation makes the deterministic document splits. Training starts with random weights and writes a new checkpoint under `artifacts/v4`.

```powershell
python download_v4_tinystories.py
python prepare_v4_tinystories.py
python train_v4_model.py
python evaluate_v4.py
```

Do not run these commands just to use the saved demo. Training needs the downloaded source and prepared documents; inference does not. The active V4 artifact paths are:

- Checkpoint: [`artifacts/v4/v4_best_20261121_170608_22072_1b90f1c4.pth`](artifacts/v4/v4_best_20261121_170608_22072_1b90f1c4.pth)
- Matching tokenizer: [`artifacts/v4/tokenizer_v4_20261121_170608_22072_1b90f1c4.json`](artifacts/v4/tokenizer_v4_20261121_170608_22072_1b90f1c4.json)
- Inference loader and generation: `v4_inference.py`

## Repository map

| Path | Purpose |
|---|---|
| `app.py` | Existing V3 Streamlit app. |
| `streamlit_v4_app.py` | Separate cached V4 Streamlit demo. |
| `demo_v4.py`, `v4_inference.py` | Standalone V4 prompt demo and checkpoint loader. |
| `v4_model.py`, `v4_tokenizer.py`, `v4_data.py` | V4 Transformer, tokenizer, and document/window data logic. |
| `download_v4_tinystories.py`, `prepare_v4_tinystories.py`, `train_v4_model.py`, `evaluate_v4.py`, `benchmark_v4.py` | V4 data, training, benchmark, and evaluation pipeline. |
| `artifacts/v4/` | Saved V4 checkpoint/tokenizer and benchmark, training, and evaluation reports. |
| `data/v4/tinystories_manifest.json` | Dataset attribution, source checksums, and selection manifest; raw and prepared stories are excluded. |
| `language_model*.py`, `model.py`, `tokenizer.py`, `inference.py` | Earlier V1/V2 training and inference code. |
| `generation_*.py`, `high_quality_*.py`, `train_v3_model.py`, `v3_inference.py` | V2/V3 experiments and inference. |
| `tests/` | Offline unit tests for model and data pipelines. |

The repository includes model checkpoints, including the V1–V3 artifacts, but excludes raw datasets, prepared story text, local environments, and cache files. Each committed checkpoint file is below GitHub's per-file 100 MB limit.

## Tests

Run the offline unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

Tests do not train a model. The V4 saved artifact can be checked independently by running `python demo_v4.py --prompt "Once upon a time, there was a little rabbit who"`.

## Deploy the V4 demo on Streamlit Community Cloud

The repository is prepared for a separate Cloud app. Deployment still requires signing in to the user's Streamlit Community Cloud account; no public app URL is claimed until the app is created and loads successfully.

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with the GitHub account that can access `sanjanarose-pixel/NextWordPredictor`.
2. Choose **Create app** and select **Use existing repo**.
3. Set repository to `sanjanarose-pixel/NextWordPredictor`, branch to `main`, and main file path to `streamlit_v4_app.py`.
4. In **Advanced settings**, select Python 3.12, then deploy.
5. Wait for the app to finish starting. Check that an example prompt returns a continuation before sharing the URL.

Streamlit's official guide explains the repository/branch/entrypoint fields and Python selection: [deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). The app runs on CPU and only loads the committed checkpoint and tokenizer.
