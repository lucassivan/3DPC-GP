# 3DPC-GP: Probabilistic Compressive Strength Prediction of 3D Printed Concrete

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange.svg)]()

A probabilistic machine learning framework for predicting the compressive strength of 3D printed concrete (3DPC) using Gaussian Process Regression with Automatic Relevance Determination.

---

## Overview

This repository contains the core implementation for the paper:

> **"Probabilistic Compressive Strength Prediction of 3D Printed Concrete via Gaussian Process Regression: Uncertainty Decomposition and Reliability Assessment"**

Extrusion-based 3D printed concrete introduces narrow printability constraints that make systematic mix design optimisation resource-intensive. This framework provides a probabilistic, data-driven alternative: a GP model that delivers a full predictive distribution rather than a point estimate, enabling uncertainty-aware mix design and reliability-based strength assessment.

### Key Features

- **GP with ARD-RBF kernel** trained on 254 records compiled from 24 experimental programmes
- **Uncertainty decomposition** separating epistemic (reducible) and aleatoric (irreducible) components at every prediction
- **Probabilistic calibration** with verified interval coverage (PICP₉₀ = 0.920, PICP₉₅ = 0.929)
- **Reliability index maps** over the W/B × OPC design space using the fib Model Code β framework
- **GP-SHAP and ARD feature attribution** with convergent importance rankings
- **Benchmarked against 7 deterministic baselines** (ElasticNet, SVR, RF, Extra-Trees, XGBoost, LightGBM, CatBoost)
- **Interactive GUI** for real-time probabilistic prediction and reliability assessment

### Model Performance

Evaluated over 30 independent 80/20 random splits (seeds 1–30):

| Metric | GP\_ARD\_RBF (mean ± SD) |
|--------|--------------------------|
| R² | 0.887 ± 0.089 |
| RMSE (MPa) | 9.97 ± 3.50 |
| MAE (MPa) | 7.18 ± 2.18 |
| PICP₉₀ | 0.920 ± 0.043 |
| PICP₉₅ | 0.929 ± 0.038 |
| MPIW₉₀ (MPa) | 32.32 ± 3.20 |

---

## Repository Structure

```
3DPC-GP/
├── README.md
├── requirements.txt
├── gp_train.py              # Core GP training script (30-split validation + full model)
└── GUI/
    ├── gp_3dpc_gui.py       # Interactive GUI application
    ├── dataset.xlsx         # Compiled 3DPC dataset (254 records, 13 features)
    └── 3DPC-GP.png          # Application icon
```

> **Note:** The full analysis suite (calibration, uncertainty decomposition, SHAP, reliability maps, learning curves, visualisation scripts) will be released upon paper acceptance.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- CUDA-compatible GPU recommended (CPU inference is also supported)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lucassivan/3DPC-GP.git
cd 3DPC-GP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
gpytorch>=1.11.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.5.0
shap>=0.41.0
openpyxl>=3.0.0
customtkinter>=5.0.0
```

---

## Usage

### Training Script

Run the core GP training and evaluation pipeline:

```bash
python gp_train.py
```

This script will:
1. Load the 3DPC dataset from `GUI/dataset.xlsx`
2. Execute 30 independent 80/20 random splits (seeds 1–30)
3. Train a GP\_ARD\_RBF model with 5 MLL restarts per split
4. Evaluate all 8 models (GP + 7 baselines) on each split
5. Save per-split predictions, ARD length-scales, and summary metrics to `results/`

**Reproducibility:** All results are deterministic. Splits use `train_test_split(random_state=seed)` for seed ∈ {1, …, 30}; GP restarts use `torch.manual_seed(seed × 100 + r)` for r ∈ {0, 1, 2, 3, 4}.

### Graphical User Interface

Launch the interactive prediction GUI:

```bash
python GUI/gp_3dpc_gui.py
```

> The GUI requires a trained full model file (`results/gp_full_model.pt`). Run `gp_train.py` first to generate it.

The GUI allows you to:
- Input all 13 mix-design parameters with out-of-range warnings
- Obtain the GP posterior mean and 90%/95% prediction intervals instantly
- View the epistemic/aleatoric uncertainty decomposition
- Compute the reliability index β at five standard design thresholds (20, 40, 60, 80, 100 MPa)
- Load a built-in example mix (plain OPC 3DPC mortar, actual CS = 81.47 MPa)

---

## Input Parameters

The model accepts 13 mix-design features:

| Symbol | Description | Unit | Dataset Range |
|--------|-------------|------|---------------|
| f(OPC) | OPC cement grade | MPa | 0 – 52.5 |
| n(OPC) | OPC binder fraction | – | 0 – 1.00 |
| f(SAC) | SAC cement grade | MPa | 0 – 42.5 |
| n(SAC) | SAC binder fraction | – | 0 – 0.60 |
| n(FA) | Fly ash binder fraction | – | 0 – 0.75 |
| n(GGBS) | GGBS binder fraction | – | 0 – 0.30 |
| n(SF) | Silica fume binder fraction | – | 0 – 0.27 |
| n(W/B) | Water-to-binder ratio | – | 0 – 0.65 |
| n(B/S) | Binder-to-sand ratio | – | 0 – 4.50 |
| n(WRA) | WRA dosage (% by mass of binder) | % | 0 – 18.0 |
| n(Fb-E) | End-hooked steel fibre content | kg/m³ | 0 – 200 |
| n(Fb-L) | Long straight steel fibre content | kg/m³ | 0 – 16.0 |
| n(Fb) | Total fibre volume fraction | % | 0 – 4.00 |

Inputs outside the dataset range trigger a warning in the GUI; predictions in extrapolated regions carry higher epistemic uncertainty.

---

## Output Files

Running `gp_train.py` generates the following files in `results/`:

| File | Description |
|------|-------------|
| `gp_per_split_preds.csv` | Per-split predictions with epistemic/aleatoric uncertainty |
| `gp_ard_lengthscales.csv` | ARD length-scale mean ± SD across 30 splits |
| `gp_metrics_summary.csv` | R², RMSE, MAE, PICP, ECE, NLL for all 8 models |
| `gp_full_model.pt` | Trained GP model (full dataset) — required by the GUI |

---

## Code Availability

### Currently Available

- ✅ **Core GP training script** (`gp_train.py`)
- ✅ **GUI application** (`GUI/gp_3dpc_gui.py`)
- ✅ **Compiled 3DPC dataset** (`GUI/dataset.xlsx`, 254 records, 24 sources)

### Coming Soon (upon paper acceptance)

- ⏳ Probabilistic calibration and Q-Q analysis scripts
- ⏳ Uncertainty decomposition and subgroup analysis
- ⏳ GP-SHAP feature attribution pipeline
- ⏳ Reliability index map generation
- ⏳ Learning curve analysis
- ⏳ All figure reproduction scripts

### Code Request

For access to specific components before the full release, please contact:

📧 **Email**: lucassivan@163.com

Please include your institutional affiliation, research purpose, and the specific components needed.

---

## Dataset

The dataset contains **254 compressive strength records** compiled from **24 experimental programmes** in the peer-reviewed literature, covering:

- Mix types: plain OPC mortar, SAC-blended, SCM-blended (FA/GGBS/SF), fibre-reinforced
- Compressive strength range: 11.1 – 189.0 MPa (mean 64.1 MPa, SD 31.6 MPa)
- 13 mix-design features; process and testing variables (specimen geometry, curing condition, printing direction) were not reported consistently across sources and are absent from the predictor space

**Primary source:** Li, Z. et al. (2023). Compiled 3DPC database. [Dataset cited in manuscript as ref-9.]

---

## Experimental Environment

| Component | Specification |
|-----------|--------------|
| OS | Windows 11 |
| Python | 3.13.5 |
| PyTorch | 2.9.1 |
| GPyTorch | 1.14.3 |
| scikit-learn | 1.6.1 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.15.3 |
| matplotlib | 3.10.0 |
| SHAP | 0.50.0 |

---

## Citation

If you find this work useful, please cite our paper (BibTeX will be updated upon publication):

```bibtex
@article{3dpc_gp_2026,
  title   = {Probabilistic Compressive Strength Prediction of 3D Printed Concrete
             via Gaussian Process Regression: Uncertainty Decomposition and
             Reliability Assessment},
  journal = {Materials Today Communications},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, collaboration, or bug reports:

📧 **Email**: lucassivan@163.com

---

<p align="center">
  <i>If you find any bugs or have suggestions, please open an issue or submit a pull request.</i>
</p>
