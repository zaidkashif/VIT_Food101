import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification


HF_REPO_ID = os.getenv("HF_MODEL_REPO", "Zaidhehe/vit-food101-vit-base-patch16-224")

@st.cache_resource(show_spinner=False)
def load_bundle(repo_id: str):
    processor = AutoImageProcessor.from_pretrained(repo_id)
    model = AutoModelForImageClassification.from_pretrained(repo_id)
    with open(model.repocard_url if hasattr(model,'repocard_url') else processor.cache_dir, 'r') if False else None:
        pass  # placeholder to avoid unused variable lint

    # id2label is in config; but we also saved a file in training. Prefer config if present.
    id2label = getattr(model.config, "id2label", None)
    if not id2label or len(id2label)==0:
        # fallback: try label_maps.json collocated in HF repo (optional)
        # In practice config.id2label should be set during training init.
        id2label = {i:str(i) for i in range(model.config.num_labels)}

    # Ensure keys are ints
    id2label = {int(k): v for k, v in id2label.items()}

    size = processor.size.get("height", 224)
    eval_tfms = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, id2label, eval_tfms, device

processor, model, id2label, eval_tfms, device = load_bundle(HF_REPO_ID)
label_names = [id2label[i] for i in range(len(id2label))]

st.set_page_config(page_title="ViT Food Classification", page_icon="🍽️", layout="centered")
st.title("🍽️ ViT Food Classification (vit-base-patch16-224)")
st.caption("Weights from Hugging Face • Eval: resize→center-crop(224)→normalize")

with st.sidebar:
    st.subheader("Settings")
    topk = st.slider("Top-k predictions", min_value=1, max_value=min(10, len(label_names)), value=5, step=1)
    st.write("Device:", device.upper())
    st.write("Classes:", len(label_names))

uploaded = st.file_uploader("Upload a food image (JPG/PNG)", type=["jpg","jpeg","png"])

def predict_pil(img, k=5):
    x = eval_tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x).logits, dim=-1)[0].cpu().numpy()
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
        st.bar_chart({lbl: p for lbl, p in preds})
