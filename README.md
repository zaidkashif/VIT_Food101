
# ViT Food Classification (Streamlit)

**Model**: `google/vit-base-patch16-224` fine-tuned on Food-101  
**Weights**: hosted on Hugging Face → `Zaidhehe/vit-food101-vit-base-patch16-224`

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
export HF_MODEL_REPO=Zaidhehe/vit-food101-vit-base-patch16-224
streamlit run app.py
```

## Notes
- Preprocessing: Resize (~256) → CenterCrop(224) → Normalize (ImageNet mean/std).
- Typical confusions: chocolate cake vs mousse, steak vs filet mignon, ramen vs pho.
- Model weights are not committed here; they live on Hugging Face Hub.
- If your HF repo is private, set env var `HF_TOKEN` and pass it to `from_pretrained(..., token=os.getenv("HF_TOKEN"))`.
