import streamlit as st
from PIL import Image
import io

from matcher import match

st.set_page_config(page_title="Face to Animal", page_icon="🐾", layout="centered")

st.title("🐾 Face to Animal")
st.markdown("Take a selfie and discover which animal you resemble most!")

img_data = st.camera_input("Take a selfie")

if img_data is not None:
    image = Image.open(io.BytesIO(img_data.getvalue())).convert("RGB")

    with st.spinner("Analyzing your animal spirit..."):
        results = match(image, top_k=5)

    top = results[0]
    st.markdown("---")
    st.markdown(f"## You are a **{top['animal'].title()}** {top['emoji']}")
    st.markdown(f"Confidence: `{top['score']*100:.1f}%`")

    st.markdown("### Top matches")
    for r in results:
        bar_label = f"{r['emoji']} {r['animal'].title()}"
        st.progress(r["score"], text=bar_label)
