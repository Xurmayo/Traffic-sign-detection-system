# Traffic Sign Classification

Six-class traffic sign classifier built for a Computer Vision course assignment. A simple CNN baseline, an improved CNN and a fine-tuned MobileNetV2 are trained on the same data and compared with accuracy, precision, recall, F1-score and confusion matrices.

**Classes:** `stop`, `speed_limit`, `no_entry`, `pedestrian`, `turn_left`, `turn_right`

## Results

All numbers are on a held-out test set of 86 images (used once, after training).

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Baseline CNN | 0.767 | 0.799 | 0.735 | 0.691 |
| Improved CNN | 0.814 | 0.817 | 0.799 | 0.797 |
| **MobileNetV2 (transfer learning)** | **0.884** | **0.865** | **0.845** | **0.848** |
| Transfer model trained on own photos only | 0.209 | 0.367 | 0.234 | 0.216 |

The last row trains on the self-collected photos and their augmented copies only, then tests on external images. It shows how much the models depend on the variety of the training data.

The test set is small (9 `turn_left` and 7 `turn_right` images), so a difference of a few points between models is not conclusive. The ranking baseline < improved < transfer is the reliable finding.

## Repository structure

```
.
├── notebooks/
│   ├── 02_preprocessing.ipynb      # dataset checks, augmentation, folder setup
│   └── traffic_sign_project.ipynb  # data loading, training, evaluation
├── dataset/                        # own photos, one folder per class
├── augmented/                      # augmented copies of the own photos
├── external/                       # supplementary images, one folder per class
└── README.md
```

`models/`, `results/` and `report/` are not included in the repository. The notebook creates `models/` and `results/` and fills them when it runs.

## Dataset

| Folder | Images | Per class | Used for |
| --- | --- | --- | --- |
| `dataset/` | 29 | 4-5 | Training only |
| `augmented/` | 540 | 90 | Training only |
| `external/` | about 430 | about 35-95 | 60% train, 20% validation, 20% test |

- **`dataset/`** holds photos taken by the author.
- **`augmented/`** holds edited copies of `dataset/` (rotation, scale, shift, brightness, contrast, occasional blur). There are no horizontal flips, because flipping a `turn_left` sign gives a `turn_right` sign.
- **`external/`** holds supplementary images from the web, some from stock-photo sites. They may be copyrighted and are included for coursework only. Check the source licences before reusing them.

Own photos and their augmented copies never enter validation or test, so near-duplicates cannot leak into the test set. `external/` is split once with a fixed seed (42), stratified by class.

## Models

| Model | Description |
| --- | --- |
| Baseline CNN | 3 conv blocks (32, 64, 128 filters), flatten, dense 128, softmax |
| Improved CNN | Baseline architecture plus in-model augmentation and dropout |
| MobileNetV2 | ImageNet-pretrained backbone with a new head; head trained first, then the last 30 layers fine-tuned |

All models use 128 x 128 RGB input scaled to [0, 1], balanced class weights and early stopping.

## How to run

1. Install the dependencies:

   ```bash
   pip install numpy pandas matplotlib opencv-python scikit-learn tensorflow jupyter
   ```

2. Keep the folder layout above, because the notebooks load images from `../dataset`, `../augmented` and `../external`.
3. Open `notebooks/traffic_sign_project.ipynb` and run the cells from top to bottom. Training the three models takes a few minutes on a CPU.
4. Optionally run `notebooks/02_preprocessing.ipynb` first to regenerate `augmented/` from `dataset/`. It also contains a cell that renames the images in `dataset/`, so skip that cell if the files are already renamed.

Results can differ slightly between runs, even with fixed seeds.

## Limitations

- Only 29 photos are self-collected, below the usual target of 100-200 per class. Augmented copies add no new scenes.
- Part of the data is web-sourced, and some test images are near-identical frames of the same scene, which may make accuracy optimistic.
- Left and right turn signs are the hardest pair: they are mirror images and the smallest classes.
- Each model was trained once, with no repeated runs.

## Author

Alisher Amangeldi, Computer Vision course, 2026
