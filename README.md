# FootyPredict — Dual-Stage Machine Learning Framework for Football Forecasting

[![Python 3.14](https://img.shields.io/badge/Python-3.14-0284c7?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn 1.9](https://img.shields.io/badge/Scikit--Learn-1.9.1-10b981?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![SciPy 1.18](https://img.shields.io/badge/SciPy-1.18.1-8b5cf6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-slate?style=flat-square)](LICENSE)

> **Course:** Artificial Intelligence & Data Science II (AI & DS II)  
> **Student:** Saad Vakharia &nbsp;|&nbsp; **Roll No:** 155 &nbsp;|&nbsp; **Class:** BEIT-2 &nbsp;|&nbsp; **Batch:** B2  
> **Institution:** Department of Information Technology, Atharva College of Engineering, University of Mumbai  
> **Guide:** Prof. Suvarna Panchal  

---

## 📌 Executive Overview

Association football match forecasting represents an inherently high-entropy statistical problem characterized by low-frequency scoring events (~2.6 goals/game) and high variance. Standard standalone classifiers routinely suffer from draw-suppression bias, while standalone Poisson regressors fail to capture complex non-linear momentum dynamics.

**FootyPredict** decouples match forecasting into a synchronized **Dual-Stage Machine Learning Pipeline**:
1. **Stage 1 (Discrete Outcome Classification):** Multi-class Gradient Boosted Decision Trees (GBDT) predicting $P(\text{Home})$, $P(\text{Draw})$, and $P(\text{Away})$ calibrated with multi-class log-loss.
2. **Stage 2 (Bivariate Poisson Expectancy):** Independent Poisson intensity parameter estimation ($\lambda_h, \lambda_a$) generating the complete joint probability matrix:
   $$P(X=x, Y=y) = \frac{\lambda_h^x e^{-\lambda_h}}{x!} \times \frac{\lambda_a^y e^{-\lambda_a}}{y!}$$
3. **Convex Blended Inference:** Dual-stage probabilities are blended via $\alpha = 0.55$, mitigating class imbalance and outputting the top 5 exact scorelines with calibrated fair odds.

---

## 📊 Academic Benchmark & Evaluation Metrics

Tested on a stratified 20% hold-out test set (440 fixtures across 2,200 multi-season matches):

| Model Architecture | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (C=0.5)** | 49.32% | 32.48% | 33.33% | 32.48% | 41.75% |
| **Random Forest (150 Trees)** | 49.09% | 31.36% | 33.15% | 31.36% | 40.63% |
| **Gradient Boosted Trees (GBDT)** | 48.18% | 33.00% | 33.82% | 33.00% | 41.57% |
| **Neural Network (MLP: 64×32)** | 40.45% | 35.66% | 34.20% | 35.66% | 40.28% |
| **FootyPredict (Dual-Stage Ensemble)** | **50.45%** | **33.84%** | **34.42%** | **33.84%** | **43.09%** |

*Note: In association football, pure theoretical predictability is bounded by match entropy (~50–55% for 3-way outcomes). The dual-stage formulation produces a +1.52% improvement in weighted F1-score over standalone GBDT.*

---

## 📂 Repository Structure

```
AI_DS_II/
├── data/
│   └── matches.csv                  # 2,200 processed matches with engineered rolling features
├── models/
│   ├── dual_stage_model.pkl         # Trained dual-stage model checkpoint
│   ├── scaler.pkl                   # Fitted StandardScaler
│   └── evaluation_results.json      # Structured benchmark metrics
├── notebooks/
│   └── footypredict_pipeline.ipynb  # End-to-end Jupyter exploration & training notebook
├── src/
│   ├── generate_dataset.py          # Realistic multi-season dataset generator
│   ├── data_loader.py               # Feature extraction and train-test splitting
│   ├── model.py                     # DualStageFootballPredictor class & Poisson matrix
│   ├── train.py                     # Training execution and model persistence
│   └── evaluate.py                  # Generates academic figures & confusion matrix
├── app/
│   ├── index.html                   # High-signal dark obsidian analytics dashboard
│   ├── app.css                      # Precision CSS (4-6px radius, zero AI slop)
│   └── app.js                       # Client-side inference and simulation engine
├── report/
│   ├── figures/                     # High-resolution publication charts
│   └── FootyPredict_Mini_Project_Report.docx  # Formatted 5-page submission document
├── tests/
│   └── test_footypredict.py         # Comprehensive pytest suite (38 unit & regression tests)
├── requirements.txt                 # Exact pinned dependencies (includes pytest)
└── README.md
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/SaadVakharia/FootyPredict.git
cd FootyPredict

# Install requirements
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 3. Run Training & Benchmarking
By default, training uses a **chronological split** (earliest 80% train, latest 20% test) to prevent temporal data leakage from rolling features:
```bash
python src/train.py
```
To reproduce the legacy random-stratified benchmark split:
```bash
python src/train.py --random-split
```

### 4. Generate Evaluation Figures
```bash
python src/evaluate.py
```

### 5. Launch Interactive Web Dashboard
Simply open `app/index.html` in any modern web browser, or serve locally:
```bash
python -m http.server 8000 --directory app
```
Navigate to `http://localhost:8000`.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
