# Credit Card Fraud Detection with Decision Trees and SVM

## Overview
This notebook implements machine learning models to detect fraudulent credit card transactions using two classification algorithms: **Decision Trees** and **Support Vector Machines (SVM)**. The project demonstrates how different models respond to feature selection and dimensionality.

## Dataset
- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Size**: 284,807 transactions
- **Features**: 30 numerical features (V1-V28, Time, Amount)
- **Target**: Class (0 = Legitimate, 1 = Fraudulent)
- **Class Imbalance**: Only 0.172% fraudulent transactions (492 out of 284,807)

## Notebook Structure

### 1. **Library Installation and Imports**
```python
%pip install pandas scikit-learn matplotlib
```
**Reason**: Installs required packages for data processing, machine learning, and visualization.

**Key Libraries**:
- `pandas`: Data manipulation and loading
- `scikit-learn`: Machine learning algorithms and preprocessing
- `matplotlib`: Visualization

---

### 2. **Load Dataset**
```python
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/..."
raw_data = pd.read_csv(url)
```
**Reason**: Fetches the credit card transaction dataset from a cloud URL for analysis.

---

### 3. **Dataset Analysis**

#### A. Class Distribution Visualization
```python
labels = raw_data.Class.unique()
sizes = raw_data.Class.value_counts().values
plt.pie(sizes, labels=labels, autopct='%1.3f%%')
```
**Reason**: Displays the class imbalance problem visually, showing that fraudulent transactions are rare (0.172%). This identifies the need for balanced sample weighting during training.

#### B. Feature Correlation Analysis
```python
correlation_values = raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))
```
**Reason**: Shows which features have the strongest relationship with fraud detection. This helps identify the most important predictors.

---

### 4. **Data Preprocessing**

#### A. Standardization
```python
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
```
**Reason**: 
- Removes mean and scales features to unit variance
- Ensures all features are on the same scale (important for SVM)
- Prevents features with larger ranges from dominating the model

#### B. Feature Extraction
```python
X = data_matrix[:, 1:30]  # Exclude Time variable
y = data_matrix[:, 30]     # Target variable
```
**Reason**: 
- Excludes Time (not useful for prediction)
- Uses V1-V28 and Amount as features
- Separates features from labels

#### C. L1 Normalization
```python
X = normalize(X, norm="l1")
```
**Reason**: 
- Normalizes each sample to sum to 1
- Improves model convergence
- Scales features to comparable magnitudes

---

### 5. **Train/Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
**Reason**: 
- Splits data into 70% training and 30% testing
- `random_state=42` ensures reproducibility
- Evaluates model generalization on unseen data

---

### 6. **Decision Tree Model**

#### A. Sample Weight Computation
```python
w_train = compute_sample_weight('balanced', y_train)
```
**Reason**: 
- Handles class imbalance by giving more weight to fraudulent transactions
- Prevents the model from ignoring the minority class

#### B. Model Training
```python
dt = DecisionTreeClassifier(max_depth=4, random_state=35)
dt.fit(X_train, y_train, sample_weight=w_train)
```
**Reason**: 
- `max_depth=4` limits tree depth to prevent overfitting
- `random_state=35` ensures reproducibility
- Sample weights balance the imbalanced dataset

#### C. Prediction and Evaluation
```python
y_pred_dt = dt.predict_proba(X_test)[:,1]
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
```
**Reason**: 
- `predict_proba` returns probability of fraud class
- ROC-AUC measures ability to distinguish fraud from legitimate transactions
- Higher AUC = better model

---

### 7. **Support Vector Machine Model**

#### A. Model Training
```python
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
svm.fit(X_train, y_train)
```
**Reason**: 
- `class_weight='balanced'` handles class imbalance
- `loss="hinge"` is standard for SVM classification
- `fit_intercept=False` works with normalized data
- No separate sample weights needed (built into constructor)

#### B. Prediction and Evaluation
```python
y_pred_svm = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
```
**Reason**: 
- `decision_function` returns raw scores (not probabilities)
- These scores work directly with ROC-AUC scoring

---

### 8. **Feature Selection Exercise (Q1-Q2)**

#### Q1: Find Top 6 Features
```python
correlation_values = abs(raw_data.corr()['Class']).drop('Class')
correlation_values = correlation_values.sort_values(ascending=False)[:6]
```
**Reason**: 
- Uses absolute correlation to find features most related to fraud
- Identifies: V3, V10, V12, V14, V16, V17
- Demonstrates feature engineering importance

#### Q2: Modify Features
```python
X = data_matrix[:,[3,10,12,14,16,17]]
```
**Reason**: Selects only the 6 most correlated features for retraining

---

### 9. **Decision Tree with 6 Features (Q3)**

```python
X = normalize(X, norm="l1")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# ... retrain Decision Tree with 6 features
print('Decision Tree ROC-AUC score with 6 features: {0:.3f}'.format(roc_auc_dt))
```
**Key Finding**: **ROC-AUC INCREASES** with feature selection
- Decision Trees benefit from focusing on most important features
- Reduces noise from less important variables
- Improves generalization

---

### 10. **SVM with 6 Features (Q4)**

```python
# ... retrain SVM with 6 features
print('SVM ROC-AUC score with 6 features: {0:.3f}'.format(roc_auc_svm))
```
**Key Finding**: **ROC-AUC DECREASES** with feature selection
- SVM relies on high-dimensional feature space
- Creates more complex decision hyperplanes
- Loses discriminative power with fewer features

---

### 11. **Summary and Inferences (Q5)**

```python
print("=" * 60)
print("INFERENCES: Decision Trees vs SVMs")
print("=" * 60)
print("\n1. Feature Selection Impact:")
print("   - Decision Tree: ROC-AUC INCREASED with 6 selected features")
print("   - SVM: ROC-AUC DECREASED with 6 selected features")
print("\n2. Model Characteristics:")
print("   - Decision Trees benefit from feature selection")
print("   - SVMs require higher feature dimensionality")
```

---

## Key Insights

### Decision Trees
✅ **Strengths**:
- Interpretable decision rules
- Automatically performs feature selection
- Works well with fewer, important features
- Handles non-linear relationships naturally

❌ **Weaknesses**:
- Can overfit with too many features
- Sensitive to noise in irrelevant features

### Support Vector Machines
✅ **Strengths**:
- Works well in high-dimensional spaces
- Effective with many features
- Finds optimal separation hyperplane
- Robust to outliers

❌ **Weaknesses**:
- Performance degrades with feature reduction
- Less interpretable than Decision Trees
- Requires proper scaling/normalization

---

## How to Run

1. **Load the notebook** in Jupyter or VS Code
2. **Run cells sequentially** to load data and train models
3. **Execute Q1-Q2** to perform feature selection
4. **Run Q3-Q4** to compare model performance
5. **Review Q5** for final conclusions

---

## Performance Metrics

- **ROC-AUC Score**: Measures the ability to distinguish fraudulent from legitimate transactions across all probability thresholds
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Interpretation**: Higher is better

---

## Conclusion

This notebook demonstrates that:
1. Different ML algorithms respond differently to feature engineering
2. Feature selection improves Decision Tree performance
3. SVM requires higher dimensionality for optimal results
4. Handling class imbalance is crucial for fraud detection
5. Proper preprocessing (scaling, normalization) is essential

The choice between Decision Trees and SVM depends on:
- **Use Case**: Interpretability vs. Performance
- **Data**: Feature count and dimensionality
- **Problem**: Class balance and sample size
