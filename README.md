# Flower Classification: EfficientNetB0 (Scratch vs Transfer Learning)

Image classification of flower species with **EfficientNetB0**, comparing a model trained from scratch against one fine-tuned from ImageNet weights. This notebook is "Model B" of a two-model comparison against VGG16.

## Dataset

- **Source:** [Kaggle:]([https://www.kaggle.com/](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition))
- **Task:** 5-class flower classification
- **Classes:** daisy, dandelion, rose, sunflower, tulip
- **Image size used:** 150x150, batch size 32

The data is **not included** in this repo. Download it from the link above and split it into `train/`, `validation/`, and `test/` folders (one sub-folder per class). The notebook expects this layout:

```
flowers_split/
├── train/
│   ├── daisy/
│   ├── dandelion/
│   ├── rose/
│   ├── sunflower/
│   └── tulip/
├── validation/
│   └── (same class folders)
└── test/
    └── (same class folders)
```

Then update the three path variables (`train_path`, `val_path`, `test_path`) at the top of the notebook to point to your folders.

## Models

Both models share the same classification head on top of EfficientNetB0:

`GlobalAveragePooling2D → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Dense(5, softmax)`

| Model | Base weights | Base layers |
|---|---|---|
| Scratch | None (random init) | Trainable |
| Transfer | ImageNet | Frozen |

**Training setup:** Adam (lr = 0.001), categorical cross-entropy, up to 30 epochs.
**Callbacks:** EarlyStopping (patience 10, restores best weights), ReduceLROnPlateau, ModelCheckpoint (best `val_loss`), and a custom stop at 98% training accuracy.

## Results

| Model | Test accuracy | Test loss | Stopped at epoch |
|---|---|---|---|
| EfficientNetB0, scratch | 62.07% | 1.03 | 17 |
| **EfficientNetB0, transfer** | **89.66%** | **0.34** | 11 |
| VGG16, scratch* | 68.74% | 0.80 | 30 |
| VGG16, transfer* | 77.70% | - | - |

\*VGG16 results come from the other half of the group project and are not reproduced in this notebook.

**Takeaways**
- Transfer learning gives a large gain: about +27 points of accuracy for EfficientNetB0.
- Training from scratch struggles on this small dataset; the weakest class was rose (recall 0.20).
- With pretrained weights, EfficientNetB0 beats VGG16 and has balanced precision/recall (~0.9) across classes.

## Run it

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
jupyter notebook ENB0_Final_ver1.ipynb
```

**Requirements:** `tensorflow`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

A GPU is recommended but not required.

## Files

- `ENB0_Flowers_Classification.ipynb`: full pipeline (data loading, both models, evaluation, comparison plots)

## Notes

- The notebook saves the best checkpoint to `best_model.keras`. Both models write to the same filename, so the transfer run overwrites the scratch run.
