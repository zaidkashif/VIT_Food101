import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification

HF_REPO_ID = os.getenv("HF_MODEL_REPO", "Zaidhehe/vit-food101-vit-base-patch16-224")

@st.cache_resource(show_spinner=False)
def load_bundle(repo_id: str):
    processor = AutoImageProcessor.from_pretrained(repo_id)
    model = AutoModelForImageClassification.from_pretrained(repo_id)
    # id2label should be in config set during training
    id2label = getattr(model.config, "id2label", None) or {i: str(i) for i in range(model.config.num_labels)}
    id2label = {int(k): v for k, v in id2label.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, id2label, device

processor, model, id2label, device = load_bundle(HF_REPO_ID)
label_names = [id2label[i] for i in range(len(id2label))]

st.set_page_config(page_title="ViT Food Classification", page_icon="🍽️", layout="centered")
st.title("🍽️ ViT Food Classification (vit-base-patch16-224)")
st.caption("Weights from Hugging Face • Inference uses the model’s own image processor")

with st.sidebar:
    st.subheader("Settings")
    topk = st.slider("Top-k predictions", 1, min(10, len(label_names)), 5, 1)
    st.write("Device:", device.upper())
    st.write("Classes:", len(label_names))

uploaded = st.file_uploader("Upload a food image (JPG/PNG)", type=["jpg","jpeg","png"])


def predict_pil(img, k=5):
    # Use HF processor to do resize/center-crop/normalize → returns PyTorch tensors
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # shape [1, 3, 224, 224]
    with torch.no_grad():
        logits = model(pixel_values).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    idx = probs.argsort()[-k:][::-1]
    return [(id2label[i], float(probs[i])) for i in idx]

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img, caption="Uploaded", use_container_width=True)
    with c2:
        st.subheader("Predictions")
        preds = predict_pil(img, k=topk)
        for lbl, p in preds:
            st.write(f"**{lbl}** — {p*100:.2f}%")
