# Task 1: Random Forest Classifier — Customer Churn Prediction

**Level 3 (Advanced) — Codveda Technologies Machine Learning Internship**

## Description
Implement a Random Forest model for classification on a complex dataset — customer churn prediction, using the same telecom dataset as the Logistic Regression task for direct comparison.

## Dataset
Telecom Churn dataset — 2,666 training samples, 667 test samples, 69 features. No feature scaling needed (tree-based ensemble method).

## Steps
1. Loaded and preprocessed the dataset (same pipeline as Logistic Regression task)
2. Trained a baseline `RandomForestClassifier`
3. Tuned hyperparameters using `GridSearchCV` (n_estimators, max_depth, min_samples_split), optimizing for **recall** due to class imbalance
4. Evaluated using cross-validation and classification metrics (precision, recall, F1-score)
5. Performed feature importance analysis

## Results

| Metric | Baseline | After Tuning |
|---|---|---|
| Accuracy | 94.6% | 95.1% |
| Recall (churn) | 63% | **66%** |
| Precision (churn) | 98% | 98% |

**Best hyperparameters:** `max_depth=None, min_samples_split=5, n_estimators=100`

**5-fold cross-validation recall scores:** [0.628, 0.519, 0.714, 0.628, 0.667] → mean = 0.631

### Top predictors (feature importance)
| Feature | Importance |
|---|---|
| Total day minutes | 13.8% |
| Total day charge | 11.9% |
| Customer service calls | 11.9% |
| International plan | 8.7% |

## Conclusion

A Random Forest classifier was trained on the Telecom Churn dataset to predict customer churn, using the same 69 features as the Logistic Regression task.

The baseline model already outperformed Logistic Regression significantly — accuracy improved from 85.9% to 94.6%, and recall for the churn class jumped from 25% to 63%. This is because Random Forest, as an ensemble of decision trees, can capture non-linear relationships and feature interactions that a linear model like Logistic Regression cannot.

Hyperparameter tuning via `GridSearchCV` (optimizing for recall, given the class imbalance) found the best configuration to be `max_depth=None, min_samples_split=5, n_estimators=100`, slightly improving test recall to 66% while maintaining 98% precision. 5-fold cross-validation showed recall scores ranging from 52% to 71%, reflecting some variability due to the limited number of churn samples in the dataset.

Feature importance analysis revealed that **Total day minutes**, **Total day charge**, and **Customer service calls** were the most influential predictors — differing from Logistic Regression's top features, since Random Forest captures interaction effects rather than isolated linear relationships.

**Key takeaway:** Ensemble methods like Random Forest can substantially outperform linear models on complex, imbalanced classification tasks by capturing non-linear patterns — but recall still isn't perfect (66%), showing that further techniques (e.g., class weighting, SMOTE, or threshold tuning) could still improve real-world churn detection.

## Tools
Python, scikit-learn, pandas, matplotlib

# Task 3: Neural Network (TensorFlow/Keras) — Customer Churn Prediction

**Level 3 (Advanced) — Codveda Technologies Machine Learning Internship**

## Description
Build a simple feed-forward neural network using TensorFlow/Keras for classification — customer churn prediction, using the same telecom dataset as the Logistic Regression and Random Forest tasks for a three-way model comparison.

## Dataset
Telecom Churn dataset — 2,666 training samples, 667 test samples, 69 features. Features standardized using `StandardScaler` (required for gradient-based training).

## Architecture
```
Input Layer     (69 features)
Dense (32, ReLU)
Dense (16, ReLU)
Dense (1, Sigmoid)   → binary classification output
```
Total trainable parameters: 2,753

## Steps
1. Loaded and preprocessed the dataset, standardized features
2. Designed and compiled the network (Adam optimizer, binary crossentropy loss)
3. Trained for 50 epochs — model overfit significantly (train recall 96.6% vs. validation recall 50.5%)
4. Visualized training vs. validation loss/accuracy to diagnose overfitting
5. Rebuilt the model with **Dropout (0.3)** layers and **EarlyStopping** (patience=5, restore best weights)
6. Re-evaluated on the test set

## Results

| Model | Accuracy | Recall (churn) | Precision (churn) |
|---|---|---|---|
| Initial (50 epochs, no regularization) | 88% | 52% | 60% |
| **Regularized (Dropout + EarlyStopping, stopped at epoch 33)** | **89%** | 43% | **71%** |

### Three-model comparison (same Churn dataset)
| Model | Accuracy | Recall (Churn) | Precision (Churn) |
|---|---|---|---|
| Logistic Regression | 85.9% | 25% | 51% |
| **Random Forest** | **95.1%** | **66%** | **98%** |
| Neural Network (regularized) | 89% | 43% | 71% |

## Conclusion

A feed-forward neural network was built using TensorFlow/Keras to predict customer churn, using the same 69 features as the earlier Logistic Regression and Random Forest tasks.

The initial model (2 hidden layers, 50 epochs, no regularization) showed clear overfitting: training accuracy reached 99.4% while validation accuracy plateaued around 87%, and test recall for the churn class was only 52% despite 96.6% recall on training data. The training vs. validation loss plot confirmed this — validation loss reached its minimum around epoch 12–15, then increased steadily while training loss kept falling.

To address this, a second model was built with **Dropout layers (0.3)** and **EarlyStopping** (monitoring validation loss, patience=5). Training stopped automatically at epoch 33, restoring the best weights. This improved precision for the churn class (60% → 71%) but slightly reduced recall (52% → 43%), reflecting a shift toward more conservative predictions.

Comparing all three models built on this dataset, **Random Forest outperformed the Neural Network** on every metric. This is a common finding for small-to-medium sized tabular data: tree-based ensemble methods often outperform neural networks, which typically need much larger datasets to fully leverage their capacity. Neural networks tend to shine more on unstructured data (images, text, audio) rather than structured tabular data like this churn dataset.

**Key takeaway:** Model choice should be guided by data characteristics, not just complexity. Regularization techniques (dropout, early stopping) are essential for neural networks to generalize well, but even with proper regularization, simpler ensemble methods can outperform deep learning on smaller tabular datasets.

## Tools
Python, TensorFlow/Keras, pandas, matplotlib
