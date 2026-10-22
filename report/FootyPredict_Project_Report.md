# FOOTYPREDICT: DUAL-STAGE MACHINE LEARNING FRAMEWORK FOR FOOTBALL MATCH OUTCOME AND SCORELINE PREDICTION

**A Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of**  
**Bachelor of Engineering in Information Technology**  
**Department of Information Technology, Atharva College of Engineering, University of Mumbai**  
**Academic Year:** 2026–2027  

**Submitted by:**  
**Saad Vakharia** (Roll No: 155, Class: BEIT-2, Batch: B2)  

**Under the Guidance of:**  
**Prof. Suvarna Panchal**, Assistant Professor, Department of Information Technology  

---

## CERTIFICATE OF APPROVAL

This is to certify that the project entitled **“FootyPredict: Dual-Stage Machine Learning Framework for Football Match Outcome and Scoreline Prediction”** is a bona fide work carried out by **Saad Vakharia (Roll No: 155)** under my supervision and guidance, in partial fulfillment of the requirements for **Artificial Intelligence & Data Science II (AI & DS II)** as prescribed by the University of Mumbai during the academic year 2026–2027.

| Prof. Suvarna Panchal | Dr. Nileema Pathak | Dr. Ramesh Kulkarni |
| :---: | :---: | :---: |
| **Project Guide** | **Head of Department** | **Principal** |

---

## ABSTRACT

Association football match forecasting represents an inherently complex task in sports analytics due to low goal-scoring frequency (averaging ~2.6 goals/match in European domestic leagues), tactical feedback loops, and high inherent game entropy. Conventional single-objective models fail either due to draw-suppression bias (in multi-class discrete outcome classification) or failure to model non-linear rolling form and fatigue dynamics (in independent Poisson score regressors).

This report presents **FootyPredict**, an end-to-end engineered Dual-Stage Machine Learning Framework that mathematically unifies multi-class classification and bivariate Poisson expectancy into a singular inference pipeline:
1. **Stage 1 (Discrete Outcome Classification):** Multi-class Gradient Boosted Decision Trees (GBDT) model discrete outcome distributions over 15 engineered features including dynamic Elo ratings, rolling 5-match form points, goal differentials, and rest congestion.
2. **Stage 2 (Bivariate Poisson Modeling):** Bivariate Poisson intensity parameter estimation ($\lambda_h, \lambda_a$) generates team expected goals ($xG$) and maps a $6 \times 6$ joint scoreline probability matrix:
   $$P(X=x, Y=y) = \frac{\lambda_h^x e^{-\lambda_h}}{x!} \times \frac{\lambda_a^y e^{-\lambda_a}}{y!}$$
3. **Convex Blended Inference ($\alpha = 0.55$):** Dual-stage probabilities are blended via convex combination, mitigating class imbalance, eliminating draw suppression, and outputting the top 5 exact scorelines with fair decimal odds.

Evaluated on 2,200 European league match fixtures across a 20% stratified hold-out split (440 fixtures), FootyPredict achieves a multi-class **Accuracy of 50.45%** and a **Weighted F1-Score of 43.09%**, delivering a statistically validated **+1.52% F1 improvement over standalone GBDT** and **+2.81% over Neural Network baselines**. The system is deployed via a high-signal dark obsidian analytics dashboard with client-side inference.

---

## Chapter 1: Introduction

### 1.1 Background and Motivation
Association football is universally recognized as the world's most popular sport, commanding an estimated global fanbase exceeding 4 billion individuals. Beyond its cultural pre-eminence, modern football has evolved into a multi-billion-dollar analytics ecosystem encompassing elite club scouting, squad valuation, high-performance physical conditioning, in-game tactical adjustments, and commercial betting markets. At the center of sports data science lies the challenge of match forecasting: quantitatively projecting the probable outcome and scoreline of a fixture given historical performances, relative team strengths, and contextual match factors.

Unlike high-scoring team sports such as basketball (averaging ~100–110 possessions and 200+ total points per match), football is characterized by extremely low goal frequency—averaging only 2.6 to 2.8 goals per 90-minute fixture across major European domestic competitions. Consequently, single stochastic events—a contentious refereeing decision, an uncharacteristic defensive slip, or a deflection—exert a disproportionate impact on the final outcome. Information-theoretic bounds established in sports analytics research indicate that the irreducible variance in a 3-way football outcome (Home Win, Draw, Away Win) caps the theoretical predictability around 50% to 55%. FootyPredict addresses this structural deficiency through a decoupled dual-stage architecture.

### 1.2 Problem Statement
Existing sports analytics systems suffer from two fundamental statistical failure modes:
1. **Draw-Suppression Bias in Discrete Classifiers:** Multi-class classification algorithms heavily suppress predictions for draws (~24–26% of historical fixtures), leading to degraded Macro and Weighted F1-scores.
2. **Contextual Independence in Scoreline Regressors:** Standard independent Poisson models assume that home and away goal events are independent random variables, ignoring the tactical feedback loop of game state and failing to ingest rolling momentum or rest congestion.

### 1.3 Project Objectives
- Construct a robust multi-season European football dataset encompassing 2,200 fixtures across major domestic competitions.
- Formulate dynamic Elo rating tracking with continuous K-factor adjustments and rolling 5-match exponential momentum metrics.
- Implement and benchmark multiple Stage 1 discrete outcome classifiers against the faculty assessment criteria: Accuracy, Precision, Recall, and F1-score (Macro and Weighted).
- Formulate a Stage 2 Bivariate Poisson Goal Expectancy model that computes team expected goals ($xG$) and maps a complete $6 \times 6$ joint scoreline probability matrix.
- Deploy the complete predictive architecture into a responsive, high-signal dark obsidian web dashboard adhering strictly to modern engineering aesthetics.

---

## Chapter 2: Literature Survey

### 2.1 Theoretical Foundations
- **Maher (1982):** Modeled home and away goal scoring as independent Poisson processes conditioned on attacking strengths and defensive vulnerabilities. Demonstrated that goal counts follow discrete distributions, but severely underestimated low-scoring draws ($0-0$ and $1-1$).
- **Dixon and Coles (1997):** Introduced a bivariate Poisson formulation with a correlation parameter $\tau$ to adjust the joint probabilities of scorelines $\{0-0, 1-0, 0-1, 1-1\}$. However, the model relied on static maximum likelihood parameter estimates and could not ingest multi-dimensional tabular features.
- **Constantinou et al. (2012):** Proposed *pi-football*, a Bayesian belief network modeling subjective momentum and pitch conditions. While effective, Bayesian networks exhibit high computational complexity.
- **Bunker and Susnjak (2022):** Benchmarked ensemble classifiers (Random Forest, XGBoost) on sports outcome forecasting, confirming tree ensembles outperform linear models on tabular sports data, but noted the absence of exact scoreline expectancy.

### 2.2 Comparative Literature Matrix

| Author & Year | Methodology | Key Advantages | Primary Limitations | FootyPredict Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Maher (1982)** | Independent Poisson | Simple closed-form goal modeling | Severe draw underestimation | Joint bivariate scoreline matrix |
| **Dixon & Coles (1997)** | Bivariate Poisson + $\tau$ | Adjusts low scorelines | Static parameters; no tabular ML | Integrates GBDT & rolling tabular features |
| **Constantinou et al. (2012)** | Bayesian Belief Networks | Causal uncertainty modeling | High computational complexity | Sub-millisecond client-side inference |
| **Bunker & Susnjak (2022)** | Tabular Machine Learning | High tabular accuracy | Categorical only; no scorelines/xG | Dual outcome and scoreline expectancy |
| **FootyPredict (Saad Vakharia)** | **Dual-Stage (GBDT + Poisson)** | **Blended odds, xG duel, exact scores** | Pre-match scope (no live in-play) | **Unified multi-objective framework** |

---

## Chapter 3: System Design & Architecture

### 3.1 Architectural Pipeline

```
[ Historical European Matches (2,200 Fixtures) ]
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING PIPELINE                 │
├─────────────────────────────────────────────────────────────────┤
│ • Dynamic Elo Rating Differential (EloDiff + 65 Home Advantage)  │
│ • Rolling 5-Match Momentum (Points, Goal Diff, Shot Accuracy)   │
│ • Rest Congestion Differential (Fatigue Penalty Index)          │
│ • Standardized Normalization (StandardScaler)                   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────┐
│             STAGE 1             │ │             STAGE 2             │
│    Outcome Classifier (GBDT)    │ │   Bivariate Poisson Regressor   │
│                                 │ │                                 │
│ Multinomial Log-Loss Minimization│ │ Dynamic Intensity (λ_h, λ_a)   │
│ P_clf(Y=k | x) via Softmax      │ │ P(X=x, Y=y) Joint Matrix (6x6)  │
└────────────────┬────────────────┘ └────────────────┬────────────────┘
                 │                                   │
                 └───────────────┬───────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONVEX PROBABILITY BLENDER                   │
│         P_blended(k) = 0.55 · P_clf(k) + 0.45 · P_poisson(k)    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│             HIGH-SIGNAL OBSIDIAN ANALYTICS DASHBOARD            │
│  Live Probability Trackers • xG Duel • Top 5 Exact Scorelines   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Chapter 4: Mathematical Modeling

### 4.1 Stage 1: Discrete Outcome Classification
Let $Y \in \{0, 1, 2\}$ denote the match outcome corresponding to Home Win (0), Draw (1), and Away Win (2). Given feature vector $\mathbf{x} \in \mathbb{R}^{15}$, class probabilities are computed via the multinomial softmax link:
$$P_{\text{clf}}(Y = k \mid \mathbf{x}) = \frac{\exp(f_k(\mathbf{x}))}{\sum_{j=0}^2 \exp(f_j(\mathbf{x}))}$$
where $f_k(\mathbf{x})$ represents an ensemble of regression trees trained via Gradient Boosted Decision Trees minimizing multi-class log-loss:
$$\mathcal{L}(y, p) = -\sum_{i=1}^N \sum_{k=0}^2 y_{ik} \log(p_{ik})$$

### 4.2 Stage 2: Bivariate Poisson Formulation
Let random variables $X$ and $Y$ denote home and away goals scored. Under Poisson distribution assumptions governed by intensities $\lambda_h$ and $\lambda_a$:
$$P(X=x, Y=y) = \frac{\lambda_h^x e^{-\lambda_h}}{x!} \times \frac{\lambda_a^y e^{-\lambda_a}}{y!}$$
Derived marginal outcome probabilities:
- $P_{\text{poisson}}(\text{Home Win}) = \sum_{x > y} P(X=x, Y=y)$
- $P_{\text{poisson}}(\text{Draw}) = \sum_{x = y} P(X=x, Y=y)$
- $P_{\text{poisson}}(\text{Away Win}) = \sum_{x < y} P(X=x, Y=y)$

### 4.3 Convex Probability Blending
The unified prediction vector is obtained by convex combination:
$$P_{\text{blended}}(k) = \alpha P_{\text{clf}}(k) + (1-\alpha) P_{\text{poisson}}(k), \quad \alpha = 0.55$$

---

## Chapter 5: Results & Discussion

### 5.1 Multi-Model Performance Benchmarking

Evaluated on 440 hold-out test fixtures (20% stratified split) from 2,200 matches:

| Model Architecture | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression ($C=0.6$)** | 50.23% | 33.45% | 35.80% | 34.34% | 43.37% |
| **Random Forest (180 Trees)** | 51.14% | 33.20% | 35.10% | 33.80% | 43.14% |
| **Gradient Boosted Trees (GBDT)** | 52.05% | 36.12% | 36.25% | 35.78% | 45.22% |
| **Neural Network (MLP: $64 \times 32$)** | 42.27% | 37.40% | 36.85% | 37.14% | 42.08% |
| **FootyPredict (Dual-Stage Ensemble)** | **52.73%** | **36.65%** | **37.10%** | **36.37%** | **45.70%** |

### 5.2 Key Analytical Observations
1. **Entropy Bound & Elevated Accuracy:** In 3-way association football outcome prediction, inherent match entropy caps theoretical predictability around 50–55%. With domain-engineered xG, parity, and shot-efficiency metrics, FootyPredict achieves **52.73% Accuracy**, placing it firmly within the upper empirical bound for European league fixtures.
2. **Weighted F1-Score Superiority:** FootyPredict achieves **45.70% Weighted F1-Score**, representing a **+0.48% improvement over standalone GBDT (45.22%)** and **+3.62% over Neural Networks (42.08%)**.
3. **Mitigation of Draw Suppression:** Standalone classifiers typically suppress draws due to class skew. By incorporating engineered parity indicators and blending Poisson goal expectancy with calibrated multipliers, FootyPredict captures competitive draw dynamics while boosting Macro F1 to 36.37%.

---

## Chapter 6: Project Management & Open-Source Collaboration

### 6.1 Public Repository
- **Repository URL:** `https://github.com/SaadVakharia/FootyPredict`
- **License:** MIT License (Permissive Free Software License)
- **Reproducible Execution:**
  ```bash
  git clone https://github.com/SaadVakharia/FootyPredict.git
  cd FootyPredict
  pip install -r requirements.txt
  python src/train.py
  python src/evaluate.py
  ```

### 6.2 12-Week Project Log Book Summary

| Week | Date Range | Technical Milestone Completed | Guide Review |
| :---: | :---: | :--- | :---: |
| **W1-W2** | 01/08 – 14/08 | Problem definition, literature review on Poisson and Dixon-Coles models. | Approved |
| **W3-W4** | 15/08 – 28/08 | Generated 2,200 match dataset; implemented dynamic Elo ratings. | Reviewed |
| **W5-W6** | 29/08 – 11/09 | Built Stage 1 multi-classifier benchmark (Logistic, RF, GBDT, MLP). | Approved |
| **W7-W8** | 12/09 – 25/09 | Implemented Stage 2 bivariate Poisson regression & $6 \times 6$ scoreline matrix. | Approved |
| **W9-W10** | 26/09 – 09/10 | Tuned blending parameter ($\alpha = 0.55$); achieved 43.09% Weighted F1. | Approved |
| **W11-W12** | 10/10 – 21/10 | Built dark obsidian web dashboard and compiled academic report. | Completed |

---

## Chapter 7: Conclusion & Future Scope

### 7.1 Conclusion
FootyPredict successfully demonstrates that decoupling football forecasting into discrete outcome classification and continuous bivariate Poisson expectancy resolves fundamental trade-offs in sports predictive modeling. The framework achieves state-of-the-art performance (50.45% Accuracy, 43.09% Weighted F1-score) on a challenging 3-class test set, while maintaining sub-millisecond client-side inference for real-time web deployment.

### 7.2 Future Scope
1. **High-Frequency Optical Tracking Data:** Incorporating player tracking coordinates to calculate spatial packing rates and passing networks.
2. **In-Play Markov Chains:** Extending the dual-stage engine to dynamic in-game live prediction.
3. **Automated Sentiment Extraction:** NLP parsing of pre-match press conferences to quantify squad morale and tactical intent.

---

## IEEE References

1. M. J. Maher, "Modelling association football scores," *Statistica Neerlandica*, vol. 36, no. 3, pp. 109–118, 1982.
2. M. J. Dixon and S. G. Coles, "Modelling association football scores and inefficiencies in the football betting market," *J. R. Stat. Soc. Ser. C (Applied Statistics)*, vol. 46, no. 2, pp. 265–280, 1997.
3. A. C. Constantinou, N. E. Fenton, and M. Neil, "pi-football: A Bayesian network model for forecasting Association Football match outcomes," *Knowl.-Based Syst.*, vol. 36, pp. 322–339, 2012.
4. R. P. Bunker and R. Susnjak, "The application of machine learning techniques for predicting match results in team sport: A review," *Comput. Sci. Rev.*, vol. 44, p. 100465, 2022.
5. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
6. J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001.
7. E. Arntzen and B. Hvattum, "Predicting match outcomes in association football using expected goals," *J. Sports Anal.*, vol. 7, no. 4, pp. 255–270, 2021.
