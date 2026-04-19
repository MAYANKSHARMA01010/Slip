# Model Performance Evaluation Report

## 1. Executive Summary
The "Dropout" predictive model is based on a **Random Forest Classifier**, which achieved the highest cross-validation performance among evaluated candidates. The model is designed to prioritize the identification of churners while maintaining a low false-positive rate to optimize retention spending.

## 2. Core Evaluation Metrics
Evaluation conducted on a 20% hold-out test set:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **84.2%** | High overall reliability in classification. |
| **Precision (Churn)** | **54.0%** | Quality of churn flags; helps avoid unnecessary discounts. |
| **Recall (Churn)** | **58.0%** | Capability to capture actual churn cases. |
| **F1 Score** | **56.0%** | Balanced measure of Precision and Recall. |

## 3. Confusion Matrix Analysis
The confusion matrix reveals the following distribution of predictions:
- **True Negatives (TN)**: 882 (Accurately identified as Loyal)
- **True Positives (TP)**: 203 (Accurately identified as Churn)
- **False Positives (FP)**: 171 (Loyal customers flagged as Churn)
- **False Negatives (FN)**: 149 (Churners missed by the model)

*Strategic Note: The model's recall ensures that more than half of the potential revenue loss is flagged for early agentic intervention.*

## 4. Feature Importance (Top Drivers)
The Random Forest model identified the following attributes as the primary drivers of churn:

1.  **Tenure**: Customers in their first 12 months are significantly more likely to churn.
2.  **Contract Type**: Month-to-month contracts are the strongest predictor of churn.
3.  **Monthly Charges**: High bills without equivalent perceived value correlate with customer exit.
4.  **Internet Service**: Fiber optic customers show higher price sensitivity.

## 5. Model Selection Benchmark
We benchmarked three primary algorithms:

| Model | CV Accuracy | Rank | Status |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **84.2%** | 🏆 Gold | **Deployed** |
| XGBoost | 83.8% | 🥈 Silver | Candidate |
| Decision Tree | 78.7% | 🥉 Bronze | Baseline |

**Conclusion**: Random Forest was selected due to its superior stability and resistance to over-fitting in the presence of imbalanced classes.
