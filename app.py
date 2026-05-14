import streamlit as st
from PIL import Image
import io

from matcher import match

st.set_page_config(page_title="Face to Animal", page_icon="🐾", layout="centered")

st.title("🐾 Face to Animal")
st.markdown("Take a selfie and discover which animal you resemble most!")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    st.markdown("[Get an API key](https://console.anthropic.com/)")

if not api_key:
    st.info("Enter your Anthropic API key in the sidebar to get started.")
    st.stop()

img_data = st.camera_input("Take a selfie")

if img_data is not None:
    image = Image.open(io.BytesIO(img_data.getvalue())).convert("RGB")

    with st.spinner("Analyzing your animal spirit..."):
        try:
            result = match(image, api_key)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    top = result["top_match"].title()
    emoji = result["top_emoji"]
    reason = result["reason"]

    st.markdown("---")
    st.markdown(f"## You are a **{top}** {emoji}")
    st.markdown(f"*{reason}*")

    st.markdown("### Runners up")
    for r in result.get("runners_up", []):
        label = f"{r['emoji']} {r['animal'].title()}"
        st.progress(r["score"], text=label)
