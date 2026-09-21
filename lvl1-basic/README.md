# Task 2: Simple Linear Regression — House Price Prediction

**Level 1 (Basic) — Codveda Technologies Machine Learning Internship**

## Description
Build a linear regression model to predict a continuous variable — house prices — using the classic Boston Housing dataset.

## Dataset
506 samples, 13 features (e.g. number of rooms, crime rate, distance to employment centers) + target `MEDV` (median home value in $1000s). No missing values.

## Steps
1. Loaded the dataset and assigned column names (raw file has no header)
2. Checked for missing values — none found
3. Split into training (80%) and testing (20%) sets
4. Standardized features using `StandardScaler` (for coefficient comparability)
5. Trained a `LinearRegression` model using scikit-learn
6. Interpreted the standardized coefficients
7. Evaluated using R² and Mean Squared Error (MSE)

## Results

| Metric | Value |
|---|---|
| R² (test set) | 0.669 |
| MSE | 24.29 |
| RMSE | ≈ 4.93 (~$4,930 average error) |

### Most influential features
| Feature | Coefficient | Effect |
|---|---|---|
| LSTAT (% lower status population) | -3.61 | Strongest negative effect |
| RM (average rooms per dwelling) | +3.15 | Strongest positive effect |
| DIS (distance to employment centers) | -3.08 | Negative effect |
| RAD (highway accessibility) | +2.25 | Positive effect |

## Conclusion

A Linear Regression model was trained on the Boston Housing dataset (506 samples, 13 features) to predict median house prices (MEDV).

The model achieved an **R² of 0.669** on the test set, meaning it explains about 67% of the variance in house prices, with an **RMSE of 4.93** (~$4,930 average prediction error).

The actual vs. predicted scatter plot shows the model performs well for mid-range house prices ($15k–$35k), but underestimates higher-priced homes (>$45k) and has a few notable outliers. This is a known limitation of the Boston Housing dataset, where MEDV values above $50k are capped, making the relationship harder to model linearly at the high end.

**Key takeaway:** Feature scaling made the coefficients directly comparable, revealing that neighborhood socioeconomic factors (LSTAT) and housing quality (RM) are the strongest drivers of price in this dataset.

## Tools
Python, pandas, scikit-learn, matplotlib, NumPy

# Task 3: K-Nearest Neighbors (KNN) Classifier — Iris Species Classification

**Level 1 (Basic) — Codveda Technologies Machine Learning Internship**

## Description
Build a KNN classifier to classify iris flowers into categories (species) based on petal and sepal measurements.

## Dataset
150 samples, 4 numerical features (sepal length/width, petal length/width), 3 balanced classes (setosa, versicolor, virginica — 50 each). No missing values.

## Steps
1. Loaded and explored the dataset
2. Split into training (80%) and testing (20%) sets using stratified sampling to preserve class balance
3. Standardized features using `StandardScaler` — essential for KNN since it relies on distance calculations
4. Trained a `KNeighborsClassifier` with K=5 as a baseline (93.3% accuracy)
5. Evaluated using confusion matrix, precision, and recall
6. Tested multiple K values (1 to 15) to find the optimal choice
7. Visualized accuracy vs. K

## Results

| K value | Accuracy |
|---|---|
| 1 | 96.67% |
| 3, 5 | 93.33% |
| **7, 9, 11, 15** | **96.67%** |

**Chosen model: K = 7** (stable, high accuracy, less prone to overfitting than K=1)

### Confusion Matrix (K=7)
```
[[10  0  0]   setosa: 10/10 correct
 [ 0 10  0]   versicolor: 10/10 correct
 [ 0  2  8]]  virginica: 8/10 correct (2 misclassified as versicolor)
```

## Conclusion

A K-Nearest Neighbors (KNN) classifier was trained on the Iris dataset to classify flowers into setosa, versicolor, and virginica.

After testing K values from 1 to 15, **K=7** was selected as the optimal choice, achieving **96.67% accuracy** on the test set. Smaller K values (K=3, K=5) performed slightly worse (93.33%) due to higher sensitivity to individual noisy points, while K=1 also reached 96.67% but is more prone to overfitting on unseen data. K values from 7 to 15 gave stable, consistent results, making K=7 a robust and reliable choice.

The confusion matrix showed that the model classified **setosa perfectly** (10/10), as its features are clearly distinct from the other two species. The only misclassifications occurred between **versicolor and virginica** (2 samples), which is expected given the natural overlap in their petal and sepal measurements.

**Key takeaway:** Feature scaling (StandardScaler) was essential for KNN since it relies on distance calculations, and testing multiple K values helped balance the trade-off between overfitting (low K) and underfitting (very high K).

## Tools
Python, pandas, scikit-learn, matplotlib
