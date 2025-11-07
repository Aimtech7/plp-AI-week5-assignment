# Understanding the AI Development Workflow
**Course:** AI for Software Engineering  
**Assignment:** Predicting 30-day Hospital Readmission Risk (Neural Network)  
**Author:** austine makwka (project prepared with ChatGPT)  
**Date:** 2025-11-07

## Project structure
- `report.pdf` — submission-ready 5–10 page report covering all parts of the assignment.
- `README.md` — this summary and answers to assignment parts.
- `main.py` — GitHub-ready Python script with preprocessing, model development (Neural Network), evaluation and a mock deployment outline.
- `diagram.png` — flowchart of the AI Development Workflow (CRISP-DM style).
- `assignment_package.zip` — zip archive containing all files.

---

## Quick summary of answers (full details are in report.pdf)

### Part 1 — Short Answer Questions
1. **Problem Definition**
   - Problem: Predict which discharged patients are likely to be readmitted within 30 days.
   - Objectives:
     1. Accurately predict 30-day readmission risk to enable targeted follow-up care.
     2. Reduce unplanned readmissions by enabling early interventions.
     3. Maintain patient privacy and fairness across demographics.
   - Stakeholders: clinicians (physicians/nurses), hospital administration, patients.
   - KPI: Area Under the Receiver Operating Characteristic Curve (AUC-ROC).

2. **Data Collection & Preprocessing**
   - Data sources: Electronic Health Records (EHR), claims data / hospital administrative records.
   - Potential bias: Historical disparities in access to care could bias the model against under-served groups.
   - Preprocessing steps: missing value handling (imputation), normalization/standardization of numerical features, categorical encoding (one-hot or embeddings), feature selection.

3. **Model Development**
   - Model: Feedforward Neural Network (Multi-Layer Perceptron).
   - Data split: 70% training, 15% validation, 15% test (stratified by readmission label).
   - Hyperparameters to tune: number of hidden layers and units, learning rate, regularization (alpha), batch size.

4. **Evaluation & Deployment**
   - Metrics: AUC-ROC and Recall (sensitivity) — recall is important to catch patients at risk.
   - Concept drift: changes over time in patient population or treatment protocols; monitor model performance, input distributions, and periodic recalibration.
   - Deployment challenge: scalability/integration with hospital EHR systems (latency, security).

### Part 2 — Case Study (Hospital readmission)
- Problem scope, data strategy, ethical concerns (privacy, fairness), preprocessing pipeline (feature engineering including prior admissions, comorbidity scores, lab trends), model selection (Neural Network), hypothetical confusion matrix and precision/recall example, deployment steps (API, EHR integration, monitoring), HIPAA compliance checklist, and optimization to address overfitting (dropout, early stopping, regularization).

### Part 3 — Critical Thinking
- Bias impact: biased training data may deny follow-up care to vulnerable groups; mitigation strategy: reweighting, fairness-aware metrics, representative sampling, and auditing.
- Trade-offs: interpretability vs accuracy — prefer more interpretable models (logistic regression, gradient-boosted trees with SHAP explanations) when clinical explainability is required; resource constraints may favor simpler models or model distillation.

### Part 4 — Reflection & Diagram
- Challenges: acquiring representative, high-quality labeled data and addressing privacy constraints.
- Improvements with more time: larger, multi-site datasets, prospective validation, and deployment with clinician-in-the-loop.
- Diagram: `diagram.png` (CRISP-DM stages visually laid out).

---

## How to run the code (`main.py`)
1. Ensure Python 3.8+ and packages: numpy, pandas, scikit-learn, matplotlib, joblib.
2. The script contains a synthetic data generator for demonstration; replace with your real EHR CSV file paths and column mapping.
3. Run:
```bash
python main.py
```
It will:
- Generate synthetic dataset (or load your data if you modify the code),
- Preprocess features,
- Train an MLP (Neural Network),
- Output evaluation metrics and a confusion matrix image,
- Save model artifact as `model.joblib`.

---

## Notes
- This package is prepared to be directly uploaded to a GitHub repo.
- `report.pdf` is the authoritative submission document; `README.md` mirrors key answers for quick review.

