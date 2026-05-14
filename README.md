# Face to Animal 🐾

Take a selfie and discover which animal you resemble most, powered by the [Claude Vision API](https://www.anthropic.com/claude) (Anthropic).

## How it works

1. Your selfie is sent to Claude's vision model.
2. Claude analyzes your facial features and returns the closest animal match with a fun explanation.
3. Top 5 matches are displayed with scores.

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

Enter your [Anthropic API key](https://console.anthropic.com/) in the sidebar when the app opens.

## Stack

- [Streamlit](https://streamlit.io) — UI + camera input
- [Claude Vision API](https://docs.anthropic.com/en/docs/build-with-claude/vision) — animal matching
- [Pillow](https://python-pillow.org) — image handling
