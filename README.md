**ViT Food Classification (Streamlit)**

Fine-tuned ViT Base (patch16-224) to classify food images.

Demo: add Streamlit Cloud URL here

Model (HF): Zaidhehe/vit-food101-vit-base-patch16-224

Quickstart
1) Create env
python -m venv .venv
macOS/Linux
source .venv/bin/activate
Windows (PowerShell)
.venv\Scripts\Activate.ps1

2) Install
pip install -r requirements.txt

3) Set model repo (public) or also set HF_TOKEN if private
export HF_MODEL_REPO=Zaidhehe/vit-food101-vit-base-patch16-224
# export HF_TOKEN=hf_xxx   # only if HF repo is private

4) Run
streamlit run app.py

What’s here

app.py — Streamlit app (upload image → top-k predictions)

requirements.txt — minimal deps

splits/ — dataset splits + label_maps.json (reproducibility)

(optional) report/ — metrics / confusion matrix / screenshots

Weights are not in this repo; they load from the Hugging Face model above.

Notes

Preprocessing uses the model’s AutoImageProcessor (resize → center-crop 224 → normalize).

If the HF repo is private, set HF_TOKEN in your environment; the app will use it automatically.
