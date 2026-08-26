##Loan Approval Prediction and Machine Learning Benchmark
Predicting loan approval decisions using supervised classification models.

This project implements an end-to-end Machine Learning pipeline designed to evaluate financial applicant data and classify loan eligibility. By training and comparing ten different algorithms—from foundational linear models to advanced boosting ensembles—this benchmark identifies the most accurate and reliable approach for automated loan approval prediction.

##Project Overview
Evaluating credit applications manually can be time-consuming and prone to human bias. The objective of this project is to build an automated data science solution that:

Cleans and standardizes raw financial records for modeling.

Benchmark ten supervised classification models under consistent training conditions.

Evaluates performance through multiple metrics, including Accuracy, Recall, F1 Score, and overfitting checks.

Identifies the key financial drivers influencing loan approval decisions.

Technical Stack
Language: Python

Data Manipulation: Pandas, NumPy

Data Visualization: Seaborn, Matplotlib

Machine Learning: Scikit-Learn, XGBoost, CatBoost, LightGBM

Data Pipeline and Preprocessing
Data Cleaning: Removed trailing whitespaces from feature names and text values.

Feature Exclusion: Dropped non-informative identifier columns to eliminate potential data leakage.

Imputation Strategy: Replaced missing numerical values using median statistics and missing categorical fields using modal values.

Encoding and Scaling: Encoded binary targets, converted categorical attributes into dummy variables using One-Hot Encoding, and standardized features using StandardScaler.

Stratified Data Splitting: Partitioned data into an 80% training set and a 20% testing set while preserving original class distributions.
## Benchmark Results

The performance metrics for all ten trained models on the test set are summarized below:

| Model | Train Accuracy | Test Accuracy | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 91.83% | 91.33% | 0.9416 | 0.9311 |
| Decision Tree | 100.0% | 98.36% | 0.9906 | 0.9869 |
| Random Forest | 98.33% | 97.78% | 0.9906 | 0.9823 |
| SVM (Linear) | 92.56% | 92.27% | 0.9416 | 0.9381 |
| SVM (RBF) | 95.58% | 94.50% | 0.9529 | 0.9556 |
| AdaBoost | 97.77% | 97.78% | 0.9887 | 0.9822 |
| XGBoost | 99.94% | 98.71% | 0.9925 | 0.9897 |
| CatBoost | 98.92% | 98.01% | 0.9925 | 0.9841 |
| KNN (k=7) | 93.56% | 89.46% | 0.9266 | 0.9162 |
| LightGBM | 100.0% | 98.71% | 0.9925 | 0.9897 |

##Key Performance Insights
Highest Accuracy and F1 Score: Both XGBoost and LightGBM demonstrated outstanding predictive power, achieving the highest test accuracy at 98.71% alongside a strong F1 Score of 0.9897.

Best Model Stability: CatBoost delivered superior generalization, maintaining a 98.01% test accuracy with less than a 0.9% variance between training and testing performance, making it highly robust against overfitting.

Important Drivers: Feature importance analysis revealed that financial reliability metrics—specifically CIBIL Score and Annual Income—serves as the most impactful predictors in determining loan approval status.

##How to Run the Project
Clone the repository:
git clone https://github.com/janabadr454/loan-approval-prediction.git
cd loan-approval-prediction

Install dependencies:
pip install xgboost catboost lightgbm pandas numpy scikit-learn seaborn matplotlib
