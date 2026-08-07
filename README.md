# A Comparative Study of Self-Supervised Learning Methods for Image Representation Learning

<p align="center">
    <img src="docs/figures/banner.png" width="900">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)

</p>

---

## Overview

This repository contains the implementation for the MSc Individual Research Project:

> **A Comparative Study of Self-Supervised Learning Methods Using a Common ResNet-18 Backbone**

The project presents a systematic comparison of five modern self-supervised learning (SSL) methods under identical experimental conditions.

Unlike many existing comparisons that evaluate methods using different architectures and datasets, this framework uses a **shared ResNet-18 backbone**, identical datasets, and a common evaluation pipeline to enable a fair comparison between SSL objectives.

---

## Research Objective

The objective of this project is to investigate how different self-supervised learning strategies influence:

- Representation quality
- Downstream image classification performance
- Training stability
- Convergence behaviour
- Computational efficiency
- GPU memory usage
- Training time

---

## Implemented Methods

The following SSL algorithms are implemented.

| Method | Learning Paradigm |
|---------|------------------|
| SimCLR | Contrastive Learning |
| BYOL | Bootstrap Self-Supervision |
| VICReg | Variance-Invariance-Covariance Regularization |
| Barlow Twins | Redundancy Reduction |
| LeJEPA | Joint Embedding Predictive Architecture |

---

## Backbone

All experiments use a common

**ResNet-18**

feature extractor to ensure a fair comparison between SSL objectives.

---

## Datasets

Supported datasets:

- CIFAR-10
- STL-10

Each dataset is evaluated using identical preprocessing and data augmentation pipelines where appropriate.

---

## Evaluation Protocol

Each model is evaluated using **Linear Probing**.

The encoder is frozen after SSL pretraining, while a linear classifier is trained using labelled images.

Evaluation metrics include:

- Top-1 Classification Accuracy
- Training Loss
- Validation Loss
- Training Time
- GPU Memory Usage
- Convergence Behaviour
- Feature Representation Quality

---

## Repository Structure

```text
ssl-comparison/

├── configs/
├── datasets/
├── checkpoints/
├── logs/
├── results/

├── models/
│   ├── backbone/
│   ├── heads/
│   ├── methods/
│   └── losses/

├── data/
├── engine/
├── utils/
├── scripts/
├── tests/

├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/USERNAME/ssl-comparison.git

cd ssl-comparison
```

Create a virtual environment

```bash
python3.11 -m venv .venv
```

Activate the environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

---

## Download Datasets

```bash
python scripts/download_datasets.py
```

Supported:

- CIFAR-10
- STL-10

---

## Training

Example:

```bash
python scripts/train.py --config configs/simclr.yaml
```

or

```bash
python scripts/train.py --method simclr
```

Available methods:

```
simclr
byol
vicreg
barlow_twins
lejepa
```

---

## Linear Probe Evaluation

```bash
python scripts/linear_probe.py --method simclr
```

---

## Compare All Models

```bash
python scripts/benchmark.py
```

This will automatically

- Train all SSL methods
- Perform linear probing
- Generate plots
- Export result tables

---

## Results

Results are automatically stored inside

```
results/

├── tables/
├── figures/
├── confusion_matrices/
├── embeddings/
└── reports/
```

---

## Future Improvements

Potential extensions include:

- Vision Transformer backbones
- DINOv2
- MAE
- ImageNet-100
- Mixed Precision Training
- Multi-GPU Support
- Distributed Training

---

## Citation

If you use this repository, please cite:

```bibtex
@mastersthesis{nizami2026,
  author = {Saif Nizami},
  title = {A Comparative Study of Self-Supervised Learning Methods Using a Common ResNet-18 Backbone},
  school = {Coventry University},
  year = {2026}
}
```

---

## Acknowledgements

This work builds upon the research contributions of:

- SimCLR
- BYOL
- VICReg
- Barlow Twins
- LeJEPA

Their original authors are fully credited in the dissertation and accompanying documentation.

---

## License

This project is released under the **MIT License**.