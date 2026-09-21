# Codveda Technologies — Machine Learning Internship

This repository contains all tasks completed during my Machine Learning Internship at **Codveda Technologies**, covering three levels of increasing complexity: Basic, Intermediate, and Advanced.

## 📁 Repository Structure

```
codveda-ml-internship/
├── level1-basic/
│   ├── linear-regression/      # House price prediction
│   └── knn-classifier/         # Iris species classification
├── level2-intermediate/
│   ├── logistic-regression/    # Customer churn prediction
│   └── decision-trees/         # Iris species classification
└── level3-advanced/
    ├── random-forest/          # Customer churn prediction
    └── neural-network/         # Customer churn prediction
```

## 📊 Task Summary

| Level | Task | Dataset | Result |
|---|---|---|---|
| 1 (Basic) | Linear Regression | Boston Housing | R² = 0.669 |
| 1 (Basic) | KNN Classifier | Iris | 96.67% accuracy |
| 2 (Intermediate) | Logistic Regression | Telecom Churn | 85.9% accuracy, 0.808 AUC |
| 2 (Intermediate) | Decision Trees | Iris | 96.67% accuracy |
| 3 (Advanced) | Random Forest | Telecom Churn | 95.1% accuracy |
| 3 (Advanced) | Neural Network (TensorFlow/Keras) | Telecom Churn | 89% accuracy |

## 🔑 Key Learnings

- **Feature scaling** is essential for distance/gradient-based models (KNN, Logistic Regression, Neural Networks), but unnecessary for tree-based models (Decision Trees, Random Forest).
- **Accuracy can be misleading** on imbalanced datasets — the Churn dataset (only 14.5% churners) required recall, precision, and AUC for a fair evaluation.
- **Simpler models can outperform complex ones.** A pruned Decision Tree (max_depth=3) beat its unpruned counterpart, and Random Forest outperformed both Logistic Regression and a Neural Network on the same churn dataset — showing that model choice should match the data, not just complexity.
- **Model interpretability matters.** Linear/Logistic Regression coefficients, Decision Tree visualizations, and feature importance analysis all help explain *why* a model makes its predictions, not just how accurate it is.

## 🛠️ Tools Used

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, matplotlib

## 👤 About

**Cahaya Tresna Afra Syaefu**
Machine Learning Intern @ Codveda Technologies

#CodvedaJourney #CodvedaExperience #FutureWithCodveda
