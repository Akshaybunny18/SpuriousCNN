# Shortcut Learning & Feature Visualization in CNNs

> **Precog Recruitment Task — Neural Networks & Deep Learning**

## Overview

This project investigates **shortcut learning** (also known as *Clever Hans* behavior) in Convolutional Neural Networks using a deliberately biased variant of the MNIST dataset called **Colored MNIST (CMNIST)**. The task was to:

1. **Build a "Cheater" model** that exploits color shortcuts to achieve high accuracy on easy (in-distribution) samples but collapses on a hard (out-of-distribution) test set.
2. **Visualize learned features** via feature-map maximization (activation maximization with total-variation regularization) to interpret what each convolutional layer encodes.

---

## Dataset: Colored MNIST (CMNIST)

Each grayscale MNIST digit is colorized:
- Each digit class (0–9) is assigned a unique dominant RGB color.
- **Training set** (`p = 0.95`): 95% of samples are painted with the digit's dominant color; 5% get a random color.
- **Hard test set**: Every sample gets a **non-dominant** random color — the color shortcut no longer helps.

This creates a controlled spurious correlation: the model can achieve very high accuracy by memorizing color → digit mapping, rather than learning actual digit shapes.

---

## Architecture

A minimal **3-layer CNN** (`SimpleCNN`) is intentionally kept small to amplify shortcut learning:

```
Input (3×28×28)
  → Conv2d(3→4, 3×3) + ReLU + MaxPool  →  4×14×14
  → Conv2d(4→6, 3×3) + ReLU + MaxPool  →  6×7×7
  → Conv2d(6→12, 3×3) + ReLU + MaxPool →  12×3×3
  → Flatten → Linear(108 → 10)
```

- **Optimizer**: Adam (lr=1e-3)
- **Loss**: CrossEntropyLoss
- **Epochs**: 2

---

## Results

| Split | Accuracy |
|---|---|
| Easy Validation (in-distribution) | **~97%** |
| Hard Test (out-of-distribution) | **~11%** |

> The ~86% accuracy **drop** is direct evidence of color-based shortcut learning — the model fails completely when the color cue is removed.

### Color Bias Proof

The model predicts **class 0** when shown a red-colored digit "1" (red being the color associated with 0 during training), demonstrating it relies on color, not shape.

---

## Feature Visualization (Probing)

Using **activation maximization** (`probing.py`): starting from random noise and iterating to find the image that maximally activates each convolutional channel. Regularized with:
- **Total-variation (TV) loss** — encourages smooth, natural-looking patterns
- **L2 weight decay** — prevents extreme pixel values

Visualizations are saved in `probes/` for all 4 channels of `c1`, `c2`, `c3`, and `fc`.

---

## Files

| File | Description |
|---|---|
| `mnist_color.py` | CMNIST dataset: colorizing MNIST with spurious correlations |
| `task1_cheater.py` | Main training + evaluation + visualizations |
| `probing.py` | Activation maximization for feature visualization |
| `check.py` | Quick dataset sanity check and visualization |
| `cheater_model.pth` | Saved trained model weights |
| `probes/` | Feature visualization images per layer/channel |

---

## Setup

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm Pillow
python task1_cheater.py   # Train and evaluate
python probing.py         # Feature visualization
```

---

## Key Takeaways

- **Shortcut learning is dangerous**: A model with 97% in-distribution accuracy can be nearly useless (~11%) out-of-distribution.
- **Feature visualization reveals the bias**: Early layers (`c1`, `c2`) encode color patches rather than edge detectors.
- **Small models overfit shortcuts faster** — the deliberate use of a tiny CNN amplifies the effect.

---

*Submitted as part of the Precog Lab recruitment task (NN/DL track), IIIT Hyderabad — 2026.*
