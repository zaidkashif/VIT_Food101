# Evaluation — ViT Food Classification (Food-101)

**Model:** `google/vit-base-patch16-224` (head re-init to 101 classes)  
**Hardware:** Kaggle T4 (16 GB), FP16 on  
**Training (main):** AdamW, lr=5e-5, wd=0.01, warmup=10%, effective batch≈64 (bs=32, grad_accum=2), epochs=3  
**Preprocessing:** Resize (~256) → CenterCrop(224) → Normalize (ImageNet mean/std) via `AutoImageProcessor`

## 1) Metrics

| Split | Accuracy | Eval loss | Notes |
|------:|---------:|----------:|------|
| Val   | **0.8297** | 0.7010 | from `trainer.evaluate(val_ds)` |
| Test  | **0.8436** | 0.6631 | from `trainer.evaluate(test_ds)` |

- Manual check matched (via `accuracy_score`).

> Full JSON saved as `report/metrics.json`.

## 2) Confusion Analysis

**Most frequent off-diagonal confusions (examples):**
- `chocolate_cake` → `chocolate_mousse` (7), and the reverse (6)
- `steak` ↔ `filet_mignon` (5/5), plus `steak` → `prime_rib` (4)
- `ramen` → `pho` (3)
- `bread_pudding` → `apple_pie` (5)
- `garlic_bread` → `pizza` (3)

Interpretation: visually similar textures/plating are hard at 224×224 with light augmentation.

> Matrix saved as `report/confusion_matrix.npy` and `report/confusion_matrix.png`.

## 3) Example Predictions (top-k)

*(sample outputs; probabilities shown as %)*

- **pancakes** → `pancakes (92.31%)`, `apple_pie (1.85%)`, `cheesecake (1.06%)`, …  
- **tacos** → `tacos (91.72%)`, `huevos_rancheros (2.14%)`, `nachos (1.64%)`, …  
- **risotto** → `risotto (72.94%)`, `gnocchi (3.26%)`, `lasagna (2.00%)`, …

> For the report, include 3–6 screenshots from the Streamlit app in `report/samples/`.

## 4) Reproducibility

- Model weights + processor: **Hugging Face** (`Zaidhehe/vit-food101-vit-base-patch16-224`)
- App + splits + report: **GitHub repo**
- Deterministic splits + label maps stored under `splits/`.

## 5) Limitations & Next Steps

- Confusions align with domain similarity (desserts, steak cuts, brothy noodle bowls).
- Potential improvements: +epochs (to 5), `ColorJitter`/`RandAugment`, mixup/cutmix, larger train resolution (if VRAM allows).

