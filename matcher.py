import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

ANIMALS = [
    "lion", "tiger", "leopard", "cheetah", "jaguar",
    "wolf", "fox", "coyote", "husky", "golden retriever",
    "poodle", "bulldog", "german shepherd", "labrador",
    "chimpanzee", "gorilla", "orangutan", "baboon",
    "polar bear", "brown bear", "giant panda", "koala",
    "horse", "zebra", "giraffe", "elephant", "rhinoceros",
    "hippo", "deer", "moose", "ram", "goat",
    "rabbit", "hamster", "squirrel", "raccoon", "skunk",
    "owl", "eagle", "parrot", "penguin", "flamingo",
    "dolphin", "seal", "otter", "cat", "lynx",
]

ANIMAL_EMOJI = {
    "lion": "🦁", "tiger": "🐯", "leopard": "🐆", "cheetah": "🐆", "jaguar": "🐆",
    "wolf": "🐺", "fox": "🦊", "coyote": "🐺", "husky": "🐶", "golden retriever": "🐶",
    "poodle": "🐩", "bulldog": "🐶", "german shepherd": "🐕", "labrador": "🐶",
    "chimpanzee": "🐒", "gorilla": "🦍", "orangutan": "🦧", "baboon": "🐒",
    "polar bear": "🐻‍❄️", "brown bear": "🐻", "giant panda": "🐼", "koala": "🐨",
    "horse": "🐴", "zebra": "🦓", "giraffe": "🦒", "elephant": "🐘", "rhinoceros": "🦏",
    "hippo": "🦛", "deer": "🦌", "moose": "🦌", "ram": "🐏", "goat": "🐐",
    "rabbit": "🐰", "hamster": "🐹", "squirrel": "🐿️", "raccoon": "🦝", "skunk": "🦨",
    "owl": "🦉", "eagle": "🦅", "parrot": "🦜", "penguin": "🐧", "flamingo": "🦩",
    "dolphin": "🐬", "seal": "🦭", "otter": "🦦", "cat": "🐱", "lynx": "🐱",
}

_model = None
_processor = None
_animal_embeddings = None


def _load_model():
    global _model, _processor, _animal_embeddings
    if _model is not None:
        return

    _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _model.eval()

    prompts = [f"a close-up photo of a {a}" for a in ANIMALS]
    inputs = _processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = _model.get_text_features(**inputs)
    _animal_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)


def match(image: Image.Image, top_k: int = 5) -> list[dict]:
    _load_model()

    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = _model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    similarities = (image_features @ _animal_embeddings.T).squeeze(0)
    scores = similarities.softmax(dim=0).numpy()

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "animal": ANIMALS[i],
            "emoji": ANIMAL_EMOJI.get(ANIMALS[i], "🐾"),
            "score": float(scores[i]),
        }
        for i in top_indices
    ]
