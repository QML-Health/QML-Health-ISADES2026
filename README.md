# A comprehensive study of quantum machine learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.39-7B5BA6.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F89939.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/Lisans-MIT-green.svg)](LICENSE)
[![ISADES 2026](https://img.shields.io/badge/ISADES-2026-C44569.svg)](#)

> All code, experimental protocols, and analysis scripts for the paper submitted to the **ISADES 2026 — International Symposium on Applied Data Engineering and Sciences.** 

This study systematically evaluates the performance of quantum machine learning (QML) on health classification tasks using **two contrasting datasets**: **Wisconsin Breast Cancer Diagnostic (WBCD)** (binary, 100% real) and **Estimation of Obesity Levels** (seven-class, 77% synthetic via SMOTE).

![Cross-Context Classical–Quantum Clarity](figures/banner_capraz_baglam.png)

*While the difference in accuracy between classical and quantum methods in the WBCD dataset is only **5.05 points** (Cohen’s d = 1.70), this gap increases to **14.68 points** (Cohen’s d = 5.13) in the obesity dataset. The Wilcoxon signed-rank test confirms classical superiority in both datasets with p = 0.0312.*

---

##  Table of Contents

- [Abstract](#-çalışmanın-özeti)
- [Key Findings](#-anahtar-bulgular)
- [Repo Structure](#-repo-yapısı)
- [Installation](#-kurulum)
- [Usage](#-kullanım)
- [Datasets](#-veri-setleri)
- [Models](#-modeller)
- [Visualizations](#-görseller)
- [Results](#-sonuçlar)
- [Citation](#-atıf)
- [Authors](#-yazarlar)
- [License](#-lisans)

---

##  Abstract

This study compares the performance of QML models across two structurally distinct health classification tasks using a common 5-fold stratified cross-validation protocol across **35 model variants** (16 WBCD + 19 obesity). The main methodological contributions proposed are:

1. **Cross-context evaluation protocol** — Measures how QML’s performance gap relative to classical baselines varies according to dataset characteristics
2. **Q-Hybrid-Q3-Plus architecture** — A two-arm hybrid structure combining Re-uploading (6 qubits, 2 blocks) and Amplitude (4 qubits, 2 layers); achieves 81.54% accuracy in the obesity task, outperforming pure encodings by 19.7 percentage points
3. **Encoding family ablation** — Systematic screening of qubit count and depth combinations across the Angle, IQP, Amplitude, and Re-uploading families
4. **Noise robustness analysis** — Robustness tests under depolarizing and bit-flip channels for p ∈ [0, 0.20]

---

## Key Findings

| Dimension | WBCD | Obesity |
|-------|:----:|:-------:|
| Classification type | Binary | 7-class |
| Sample size | 569 | 2,087 |
| Data source | 100% real | 77% SMOTE |
| **Classic champion** | SVM-RBF | XGBoost-Top10 |
| Classic CV accuracy | **0.9802 ± 0.0044** | **0.9623 ± 0.0061** |
| **Quantum champion** | VQC ReUpload-6q-3block | Q-Hybrid-Q3-Plus |
| Quantum CV accuracy | **0.9297 ± 0.0292** | **0.8154 ± 0.0227** |
| **Accuracy difference** | **5.05 points** | **14.68 points** |
| **Cohen's d (paired)** | **1.70** (large) | **5.13** (very large) |
| **Wilcoxon p (one-tailed)** | **0.0312*** | **0.0312*** |

For detailed results. → [Results](#-sonuçlar) section

---

##  Repo Structure

```
QML-Health-ISADES2026/
│
├── notebooks/                          # All notebooks
│   ├── QML_BreastCancer_ISADES2026.ipynb       # WBCD main notebook
│   ├── QML_Obesity_ISADES2026.ipynb            # Obesity main notebook
│   ├── obesity_classical_models.ipynb          # Obesity classic baseline
│   ├── obesity_quantum_models.ipynb            # Obesity pure quantum
│   └── obesity_hybrid_quantum.ipynb            # Q-Hybrid-Q3-Plus ablation
│
├── figures/                            
│   ├── banner_capraz_baglam.png        
│   ├── wbcd/                           
│   │   ├── roc_curves.png
│   │   ├── encoding_heatmap.png
│   │   ├── quantum_champion_cm.png
│   │   └── noise_robustness.png
│   ├── obesity/                        # Obesity figures
│   │   ├── q3plus_confusion_matrix.png
│   │   ├── reupload_ablation.png
│   │   ├── encoding_families.png
│   │   └── all_quantum_models.png
│   ├── shap/                           # SHAP 
│   │   ├── combined_shap.png
│   │   
│   └── stats/                          # Tests
│       └── wilcoxon_folds.png
│
├── results/                            # Results(JSON/CSV)
│       
│
├
├── LICENSE                             # MIT lisannce
└── README.md                           
```

---

##  Installation

### Requirements

- Python 3.11 or later
- 8 GB+ RAM (for quantum simulations)
- (Optional) GPU — only for XGBoost acceleration

### Step 1: Clone the repository

```bash
git clone https://github.com/QML-Health/QML-Health-ISADES2026.git
cd QML-Health-ISADES2026
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# or
venv\Scripts\activate             # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` içeriği:
```
pennylane==0.39.0
torch==2.4.0
scikit-learn==1.5.0
xgboost==2.1.0
shap==0.46.0
imbalanced-learn==0.13.0
matplotlib==3.9.0
seaborn==0.13.2
pandas==2.2.0
numpy==1.26.4
jupyter==1.0.0
```

---

## Usage

### With Google Colab (Recommended — quantum simulations do not require a GPU)

You can open and run the notebooks directly in Colab:

| Notebook | Runtime | Description |
|----------|:--------------:|----------|
| `QML_BreastCancer_ISADES2026.ipynb` | ~45 min | WBCD: 6 classical + 10 quantum models |
| `QML_Obesity_ISADES2026.ipynb` | ~120 min | Obesity: all phases + Wilcoxon |
| `obesity_hybrid_quantum.ipynb` | ~30 min | Q-Hybrid-Q3-Plus ablation |

### On a local machine
```bash
jupyter notebook notebooks/
```

### To reproduce the results

```python
# All randomness seeds are fixed
RANDOM_STATE = 42
np.random.seed(42)
torch.manual_seed(42)
```

5-fold CV splits can be reproduced using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

---

## Datasets
### 1. Wisconsin Breast Cancer Diagnostic (WBCD)

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Citation:** Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI Machine Learning Repository.
- **Number of samples:** 569
- **Number of features:** 30 (continuous)
- **Class distribution:** Malignant (37.3%) / Benign (62.7%)
- **Access:** Can be loaded directly using `sklearn.datasets.load_breast_cancer()`

### 2. Estimation of Obesity Levels Based on Eating Habits and Physical Condition

- **Source:** [Data in Brief (Palechor & De la Hoz Manotas, 2019)](https://doi.org/10.1016/j.dib.2019.104344)
- **Number of samples:** 2,087 (485 original + 1,602 SMOTE)
- **Number of features:** 16 (8 numerical + 8 categorical)
- **Number of classes:** 7 (Underweight, Normal, Overweight I-II, Obesity I-III)
- **Important note:** 77% of the dataset was synthetically generated using the SMOTE algorithm; this is discussed in detail in the [Method](docs/METHODOLOGY.md) section
---

## Models

### Classical Models (18 in total)

#### WBCD (6 models)
- Logistic Regression, SVM-RBF, Random Forest, XGBoost, KNN, MLP

#### Obesity (12 models — 16 features vs. Top-10 classification)
- In addition to the above: SVM-Linear, Naive Bayes, Decision Tree, AdaBoost, Gradient Boosting, LightGBM

### Quantum Models (total 17)

#### WBCD (10 models)
| Family | Configurations |
|------|-----------------|
| **Angle Embedding** | 4q, 6q |
| **IQP Embedding** | 4q, 6q |
| **Amplitude Embedding** | 5q (32 dimensions) |
| **Re-uploading** | 4q, 6q (1/2/3-block ablation) |
| **Quantum Kernel SVM** | 4q, 6q |

#### Obesity (7 models)
| Family | Configurations |
|------|-----------------|
| **Pure Angle** | 6q × 3 layers |
| **Pure Amplitude** | 4q × 3 layers |
| **Pure Re-uploading** | 6q × 1/2/3 blocks |
| **Q-Hybrid-Q3 (DualBranch)** | Angle-6q + Amplitude-4q parallel |
| **Q-Hybrid-Q3-Plus**  | Re-uploading-6q + Amplitude-4q parallel |

### Q-Hybrid-Q3-Plus Architecture (Original Contribution)

```
Input (16 features)
    ├── PCA-6 ──────► Re-uploading branch (6q × 2 block) ──► ⟨Z⟩ × 6
    └── Top-8 ──────► Amplitude branch (4q × 2 layers) ──► ⟨Z⟩ × 4
                                                              │
                                concat[10] ◄─────────────────┘
                                    │
                          Linear(10→64) → ReLU → Dropout(0.3)
                                    │
                              Linear(64→7) → Softmax
```

**Total trainable parameters:** 54 (quantum part) + classical head

---

## Figures

### Methods and Architectures

#### Figure 1 — System Architecture
![System Architecture](figures/system_architecture.png)

Parallel evaluation workflow for the WBCD and Obesity datasets — 35 model variants under 5-fold stratified CV.

#### Figure 2 — Q-Hybrid-Q3-Plus Circuit Diagram
![Q-Hybrid-Q3-Plus](figures/q3plus_circuit.png)

Two-arm quantum-classical hybrid architecture.

### WBCD Results

#### Figure 3 — ROC Curves
![WBCD ROC](figures/wbcd/roc_curves.png)

> **Drive path:** `MyDrive/QML_ISADES2026/gorseller/ROC_FINAL_v2_MAKALE.png`

#### Figure 4 — Encoding Family Heatmap
![WBCD Encoding](figures/wbcd/encoding_heatmap.png)

> **Drive path:** `MyDrive/QML_ISADES2026/gorseller/HEATMAP_Encoding_FINAL.png`

#### Figure 5 — Quantum Champion Confusion Matrix
![WBCD Champion CM](figures/wbcd/quantum_champion_cm.png)

> **Drive path:** `MyDrive/QML_ISADES2026/images/CM_Champion_ReUpload_6q_3block_FINAL.png`

#### Figure 6 — Noise Robustness
![WBCD Noise](figures/wbcd/noise_robustness.png)

> **File path:** `MyDrive/QML_ISADES2026/images/NOISE_FULL_SPECTRUM_Champion_ReUpload_6q_3block.png`

The VQC ReUpload-6q-3block model maintains an accuracy of 92.10% for depolarizing errors and 89.47% for bit-flip errors at p=0.20.

### Obesity Results

#### Figure 7 — Q-Hybrid-Q3-Plus Confusion Matrix (7 classes)
![Obesity Q3-Plus CM](figures/obesity/q3plus_confusion_matrix.png)

> **Drive path:** `MyDrive/QML_Obesity_ISADES2026/images/q3plus_confusion_matrix.png`

#### Figure 8 — Re-upload Depth Ablation
![Re-up Ablation](figures/obesity/reupload_ablation.png)

> **Drive path:** `MyDrive/QML_Obesity_ISADES2026/images/quantum_reupload_ablation.png`

Block 1: 51.05% → Block 2: 59.08% → Block 3: 61.83% — approximately a 5-point improvement per block.

#### Figure 9 — Encoding Family Comparison
![Encoding Families](figures/obesity/encoding_families.png)

> **Drive path:** `MyDrive/QML_Obesity_ISADES2026/images/quantum_family_comparison.png`

The 61–64% ceiling of pure encodings rises to 81.54% with the hybrid Q3-Plus (+19.7-point hybrid advantage).

#### Figure 10 — All Quantum Models
![All Quantum Models](figures/obesity/all_quantum_models.png)
> **Drive path:** `MyDrive/QML_Obesity_ISADES2026/gorseller/quantum_karsilastirma_bar.png`

### Cross-Context Analysis

#### Figure 11 — 4-Panel Cross-Context Overview (Key Visual)
![Cross-Context](figures/banner_cross-context.png)

#### Figure 12 — Wilcoxon Fold-Based Comparison
![Wilcoxon](figures/stats/wilcoxon_folds.png)

> **Drive path:** `MyDrive/QML_Obesity_ISADES2026/images/wilcoxon_fold_comparison.png`

### Explainability (SHAP)

#### Figure 13 — XGBoost SHAP Top Three Drivers
![SHAP](figures/shap/combined_shap.png)

| Dataset | Top 3 drivers | SHAP values |
|-----------|--------------|:--------------:|
| **WBCD** | worst perimeter, worst texture, worst area | 0.928, 0.866, 0.840 |
| **Obesity** | Weight, Height, Gender | 2.615, 0.475, 0.300 |

---

##  Results

### Table I — WBCD: 5-Fold CV Results for All Models

| Model | Type | CV Acc | Acc Std | F1 | AUC | Param |
|-------|:---:|:------:|:-------:|:--:|:---:|:-----:|
| **SVM-RBF** | Kl. | **0.9802** | 0.0044 | **0.9802** | **0.9947** | 200 SV |
| LR | Kl. | 0.9736 | 0.0054 | 0.9735 | 0.9947 | 31 |
| XGBoost | Kl. | 0.9736 | 0.0112 | 0.9735 | 0.9954 | ~10K |
| RF | Kl. | 0.9604 | 0.0149 | 0.9604 | 0.9908 | ~25K |
| KNN | Kl. | 0.9714 | 0.0149 | 0.9713 | 0.9929 | — |
| MLP | Kl. | 0.9385 | 0.0365 | 0.9385 | 0.9851 | ~40K |
| **VQC ReUpload-6q-3blok** | Q. | **0.9297** | 0.0292 | **0.9284** | **0.9898** | **54** |
| VQC ReUpload-6q | Q. | 0.9011 | 0.0374 | 0.8978 | 0.9804 | 36 |
| VQC ReUpload-4q | Q. | 0.8967 | 0.0330 | 0.8941 | 0.9735 | 24 |
| VQC IQP-4q | Q. | 0.8879 | 0.0213 | 0.8829 | 0.9714 | 24 |
| VQC Angle-4q | Q. | 0.8989 | 0.0189 | 0.8961 | 0.9677 | 24 |
| VQC Angle-6q | Q. | 0.8945 | 0.0226 | 0.8916 | 0.9648 | 36 |
| VQC IQP-6q | Q. | 0.8659 | 0.0566 | 0.8577 | 0.9454 | 36 |
| VQC Amplitude-5q | Q. | 0.8242 | 0.0547 | 0.8124 | 0.9282 | 30 |
| QKernel-SVM-4q | Q. | 0.8308 | 0.0226 | 0.8285 | 0.9047 | kernel |
| QKernel-SVM-6q | Q. | 0.7692 | 0.0184 | 0.7599 | 0.8609 | kernel |

### Table II — Obesity: 5-Fold Cross-Validation Results for All Models (summary)

| Model | Type | CV Acc | AUC | F1 |
|-------|:---:|:------:|:---:|:--:|
| **XGBoost-Top10** | Kl. | **0.9623 ± 0.0061** | **0.9971** | **0.9611** |
| SVM-RBF-Top10 | Kl. | 0.9521 ± 0.0104 | 0.9975 | 0.9498 |
| LR-Top10 | Kl. | 0.9431 ± 0.0093 | 0.9960 | 0.9405 |
| RF-Top10 | Kl. | 0.9395 ± 0.0106 | 0.9947 | 0.9373 |
| MLP-Top10 | Kl. | 0.9221 ± 0.0179 | 0.9947 | 0.9180 |
| **Q-Hybrid-Q3-Plus**  | Q. | **0.8154 ± 0.0227** | **0.9663** | **0.8073** |
| Q-Hybrid-Q3 (DualBranch) | Q. | 0.7711 ± 0.0347 | 0.9530 | 0.7581 |
| Q-Amplitude-4q-3L | Q. | 0.6429 ± 0.0260 | 0.9050 | 0.6283 |
| Q-ReUpload-6q-3blok | Q. | 0.6183 ± 0.0266 | 0.8845 | 0.5939 |
| Q-Angle-6q-3L | Q. | 0.6153 ± 0.0330 | 0.8838 | 0.5875 |
| Q-ReUpload-6q-2blok | Q. | 0.5908 ± 0.0150 | 0.8801 | 0.5619 |
| Q-ReUpload-6q-1blok | Q. | 0.5105 ± 0.0174 | 0.8314 | 0.4697 |

### Table III — Summary of Cross-Context Comparison

| Dimension | WBCD | Obesity |
|-------|:----:|:-------:|
| Number of classes | 2 (binary) | 7 (multiclass) |
| Data source | 100% real | 77% SMOTE |
| Classic champion | SVM-RBF | XGBoost-Top10 |
| Quantum champion | VQC ReUpload-6q-3blok | Q-Hybrid-Q3-Plus |
| Accuracy difference | **5.05 points** | **14.68 points** |
| AUC difference | 0.0049 | 0.0308 |
| Cohen's d (paired) | 1.70 (large) | 5.13 (very large) |
| Wilcoxon p (one-tailed) | 0.0312* | 0.0312* |

---

## 📝 Citation

If you use this work, please cite it as follows:

```bibtex
@inproceedings{qml_health_isades2026,
  title     = {Classical–Quantum Machine Learning on Health Data: A Cross-Context Comparative Study},
  author    = {Tevfik Metin, Atakan Yılmaz, Enes Furkan Kaya, Emine Gülmez, Assoc. Prof. Dr. Muhammet Baykara*},
  booktitle = {ISADES 2026 -- International Symposium on Applied Data Engineering and Sciences},
  year      = {2026},
  address   = {Uganda},
  publisher = {ISADES}
}
```

> **Note:** Full citation information will be updated after the paper is accepted.

---

##  Authors

- **[Tevfik Metin]** — *[affiliation, email]*
- **[Atakan Yılmaz]** — *[affiliation, email]*
- **[Enes Furkan Kaya]** — *[department, email]*
- **[Emine Gülmez]** — *[department, email]*
- **[Muhammet Baykara]** — Advisor — *[department, email]*
**Institution:** Fırat University, Faculty of Technology, Software Engineering, Elazığ, Turkey

---

##  Contributing

This repository contains the experiments from an academic study. Please use the [Issues](../../issues) section for bug reports and questions.

---

##  License

This project is distributed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

> Datasets are used under their original licenses:
> - WBCD: UCI Machine Learning Repository (CC BY 4.0)
> - Obesity: Data in Brief (CC BY 4.0)
---



<div align="center">

** If you found this repo helpful, don’t forget to star it!**

[![GitHub Stars](https://img.shields.io/github/stars/QML-Health/QML-Health-ISADES2026?style=social)](../../stargazers)

</div>
