# Face to Animal 🐾

Take a selfie and discover which animal you resemble most, powered by [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) running locally via PyTorch.

## How it works

1. Your selfie is encoded into an embedding vector using the CLIP vision encoder.
2. That vector is compared (cosine similarity) against text embeddings for ~50 animals.
3. The closest match wins.

No image is sent to any external server — everything runs locally.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The first run downloads the CLIP model (~340 MB) from Hugging Face and caches it locally.

## Stack

- [Streamlit](https://streamlit.io) — UI + camera input
- [PyTorch](https://pytorch.org) — tensor ops & inference
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — CLIP model
- [Pillow](https://python-pillow.org) — image handling
