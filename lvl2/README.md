# Task 1: Logistic Regression — Customer Churn Prediction

**Level 2 (Intermediate) — Codveda Technologies Machine Learning Internship**

## Description
Implement a logistic regression model to predict binary outcomes — whether a telecom customer will churn.

## Dataset
Telecom Churn dataset — 2,666 training samples, 667 test samples (pre-split 80/20), 69 features after preprocessing (including one-hot encoded state and encoded categorical variables). No missing values.

**Note:** the target class is imbalanced — only ~14.5% of customers churned.

## Steps
1. Loaded and preprocessed the dataset (encoded binary categorical variables, one-hot encoded `State`)
2. Standardized features using `StandardScaler`
3. Trained a `LogisticRegression` model using scikit-learn
4. Evaluated using accuracy, confusion matrix, classification report, and ROC curve
5. Interpreted model coefficients and odds ratios

## Results

| Metric | Value |
|---|---|
| Accuracy | 85.9% |
| AUC | 0.808 |
| Recall (churn class) | 25% |
| Precision (churn class) | 51% |

### Top predictors (by odds ratio)
| Feature | Odds Ratio | Effect |
|---|---|---|
| Customer service calls | 2.04 | Strongly increases churn risk |
| International plan | 1.98 | Increases churn risk |
| Voice mail plan | 0.43 | Decreases churn risk |

## Conclusion

A Logistic Regression model was trained on the Telecom Churn dataset to predict whether a customer would churn.

The model achieved 85.9% accuracy and an AUC of 0.808. However, accuracy alone was misleading here because the dataset is **imbalanced** — 85.5% of customers did not churn, so a model that always predicted "no churn" would already score ~85.5% accuracy without learning anything meaningful.

Looking at the confusion matrix and classification report revealed the real picture: **recall for the churn class was only 25%**, meaning the model missed 71 out of 95 customers who actually churned. This happened because the model, trained on mostly "no churn" examples, tends to assign lower churn probabilities overall — and with the default 0.5 decision threshold, many genuinely at-risk customers fell just below that cutoff and were classified as "not churning."

The AUC of 0.808, however, showed the model **does have good discriminative ability** — it generally assigns higher churn probabilities to customers who actually churn. This suggests the issue isn't the model's underlying signal, but rather the fixed 0.5 threshold being poorly suited to imbalanced data.

Interpreting the coefficients (via odds ratios) showed that **Customer service calls** was the strongest churn driver — customers who call support more often are more likely to leave. **International plan** subscribers were also more likely to churn, while **Voice mail plan** subscribers were less likely to churn, suggesting they may be more engaged/loyal customers.

**Key takeaway:** For imbalanced classification problems like churn prediction, accuracy alone is not a reliable metric. Recall, precision, and AUC give a more honest picture of model performance.

## Tools
Python, pandas, scikit-learn, matplotlib
