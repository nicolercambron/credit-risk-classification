# credit-risk-classification

Purpose of the Analysis:
The purpose of this analysis is to develop and evaluate machine learning models to predict the creditworthiness of borrowers based on historical lending activity from a peer-to-peer lending company. The model aims to assist in identifying high-risk borrowers (loan status of 1) versus healthy borrowers (loan status of 0), to mitigate financial risks for lenders.

Financial Information:
The data contains various financial attributes of borrowers, and the goal is to predict whether a loan will be "high-risk" or "healthy." The dataset includes thousands of loan records with various features such as income, loan amount, interest rate, and credit score, among others.

Variables:
The primary variable of interest is the loan_status, which indicates whether a borrower is high-risk (1) or has a healthy loan (0). The distribution of this target variable can be checked with the value_counts() function to ensure a balance between the two classes:
lending_data_df['loan_status'].value_counts()

Machine Learning Process:
The following steps were taken during the analysis:

Data Preprocessing: The data was loaded, and the target variable (loan_status) was separated from the feature set.
Train-Test Split: The dataset was split into training and testing sets using an 80-20 split to ensure sufficient data for both model training and evaluation.
Model Selection: A Logistic Regression model was chosen as the initial model for classification. This model was trained using the training data and evaluated on the testing data.
Model Evaluation: The model's performance was assessed using a confusion matrix, precision, recall, F1-score, and overall accuracy.
Methods Used:
Logistic Regression: A linear model that estimates the probability of a loan being high-risk or healthy. This model is suitable for binary classification problems like this one.
Performance Metrics: The confusion matrix and classification report provided insight into the model’s ability to predict both healthy and high-risk loans.

Results
Machine Learning Model 1: Logistic Regression
Accuracy: 99%
Precision for Class 0 (Healthy Loans): 1.00 (The model predicted healthy loans almost perfectly.)
Precision for Class 1 (High-Risk Loans): 0.86 (About 14% of predicted high-risk loans were actually healthy.)
Recall for Class 0 (Healthy Loans): 0.99 (99% of actual healthy loans were correctly identified.)
Recall for Class 1 (High-Risk Loans): 0.94 (94% of actual high-risk loans were correctly identified.)
F1-Score for Class 0: 1.00
F1-Score for Class 1: 0.90
The confusion matrix breakdown showed:

True Negatives (TN): 14,924 healthy loans were correctly identified.
False Positives (FP): 77 loans incorrectly identified as healthy.
False Negatives (FN): 31 loans incorrectly labeled as high-risk.
True Positives (TP): 476 high-risk loans were correctly classified.

Model Performance:
The Logistic Regression model performs exceptionally well for identifying healthy loans, with a precision of 1.00 and recall of 0.99.
It also performs reasonably well for identifying high-risk loans, with a precision of 0.86 and recall of 0.94. The F1-score for high-risk loans (0.90) suggests a good balance between precision and recall.
Overall, the model's accuracy is 99%, which is excellent.
Recommendation:
The Logistic Regression model is recommended for use, particularly when the goal is to accurately identify healthy loans. This is crucial for avoiding false positives, where a healthy loan might be incorrectly classified as high-risk.
If identifying high-risk loans is more critical (i.e., to avoid defaults), the model's recall for class 1 is 0.94, which means it successfully captures most high-risk loans. However, there is still room for improvement in precision for class 1, as 14% of high-risk predictions were actually healthy.
Considerations:
The importance of precision versus recall depends on business priorities. For instance, if it is more important to flag as many high-risk loans as possible, recall for class 1 is key. However, if it is more important to minimize false alarms and avoid misclassifying healthy loans, precision for class 0 should be prioritized.
This model provides a strong foundation for predicting loan risk, though future iterations could focus on improving the precision for high-risk loans, perhaps through hyperparameter tuning or exploring alternative models like Random Forests or Gradient Boosting.
