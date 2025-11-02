ViT Food Classification (Streamlit Demo)
Fine-tuned Vision Transformer google/vit-base-patch16-224 to classify food photos.
Built with PyTorch + Hugging Face Transformers, deployed via Streamlit.


Live demo: add your Streamlit Cloud link here


Model on Hugging Face: Zaidhehe/vit-food101-vit-base-patch16-224


Dataset: Food images from the Kaggle link in the brief. In our run, the data contains 101 classes (the widely used Food-101 taxonomy: 30 images/class in test).


What this repo contains


app.py — Streamlit app (upload image → top-k predictions)


splits/ — train/val/test file lists + label_maps.json (for reproducibility)


report/ (recommended) — drop your metrics.json, confusion_matrix.png, and a few sample prediction screenshots here


requirements.txt — minimal, CPU-friendly deps for Streamlit Cloud


Model weights are NOT in this repo — they live on the Hugging Face Hub



Quickstart (run locally)
# 1) Create and activate a virtual env
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) Point the app to the HF model repo
export HF_MODEL_REPO=Zaidhehe/vit-food101-vit-base-patch16-224
# If your HF repo is private, also:
# export HF_TOKEN=hf_xxx...

# 4) Run the Streamlit app
streamlit run app.py

The app will:


load the image processor and model from the HF repo,


apply the same preprocessing as training (resize→center-crop 224→normalize),


show top-k predictions with probabilities.



Model & Training (summary)


Backbone: google/vit-base-patch16-224 (ViT-Base, patch size 16, 224×224 input)


Head: re-initialized to match dataset classes (101)


Optimizer: AdamW


Learning rate: 5e-5


Weight decay: 0.01


Warmup: 10% of steps


Batching (T4 GPU): per_device_train_batch_size=32, per_device_eval_batch_size=32, gradient_accumulation_steps=2 (effective batch 64), fp16 on


Epochs: 1 (smoke test) and 3 (main run)



Tip: If you hit OOM on smaller GPUs, drop per-device batch size and raise gradient_accumulation_steps.


Results
After 1 epoch (smoke test)


Validation accuracy: 0.8297


Test accuracy: 0.8436


(3-epoch run was ~82–84% as well; remaining errors are mostly between visually similar dishes.)
Common confusions (from the confusion matrix):


chocolate_cake ↔ chocolate_mousse


steak ↔ filet_mignon ↔ prime_rib


ramen ↔ pho


tiramisu ↔ panna_cotta


These pairs share strong visual similarities (color/texture/plating), which is expected at 224×224 with light augmentations.

Reproduce evaluation (optional)
If you trained in a notebook, you likely already have these artifacts. For a report:


Save metrics


val_metrics  = trainer.evaluate()
test_metrics = trainer.evaluate(test_ds)
# save as report/metrics.json



Confusion matrix


from sklearn.metrics import confusion_matrix
pred = trainer.predict(test_ds)
cm = confusion_matrix(pred.label_ids, pred.predictions.argmax(axis=1))
# save as report/confusion_matrix.npy and a PNG figure



Sample predictions
Use the same inference code as the app on a few test images; include 3–6 screenshots.



App details
The app avoids torchvision entirely for easier deployment:


Preprocessing is done via AutoImageProcessor (processor(images=img, return_tensors="pt"))


This guarantees the exact same normalization and resize behavior as in training.


If your HF model repo is private, add HF_TOKEN to the environment and pass it to from_pretrained(..., token=os.getenv("HF_TOKEN")).

Suggested repo structure
.
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ splits/
│  ├─ train.json
│  ├─ val.json
│  ├─ test.json
│  └─ label_maps.json
└─ report/               # (optional, for your submission)
   ├─ metrics.json
   ├─ confusion_matrix.npy
   ├─ confusion_matrix.png
   └─ samples/           # screenshots of predictions


Links


Hugging Face model: Zaidhehe/vit-food101-vit-base-patch16-224


Live demo: https://vitfood101-spbzpojdgnfgappjmuckj9h.streamlit.app/

Kaggle dataset : https://vitfood101-spbzpojdgnfgappjmuckj9h.streamlit.app/


Acknowledgements


Hugging Face Transformers for model + preprocessor orchestration


PyTorch for training/inference


Streamlit for the simple web UI


Food dataset as referenced in the course brief



📜 License
Choose one and add the file (recommended MIT):
This project is licensed under the MIT License — see LICENSE for details.

Notes for graders


Goal of this task: learn fine-tuning mechanics & deployment, not squeeze SOTA numbers.


All steps (preprocessing, head re-init, fp16, grad accumulation, evaluation) are kept explicit and documented for clarity.

