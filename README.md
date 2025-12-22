**Probability-Based Classification with Threshold Optimization (Spam Detection)**

🔹**Problem Overview**

In real-world problems like spam detection, fraud detection, and risk modeling, datasets are often imbalanced. Using accuracy alone can be misleading.

This project demonstrates a production-style machine learning pipeline for binary classification (Spam vs Ham) with a strong focus on probability-based decision making rather than naive accuracy-based classification.
The key idea is to predict probabilities, tune decision thresholds, and evaluate models using business-relevant metrics such as Precision, Recall, and ROC-AUC.

🔹 Dataset

Text-based dataset (spam vs ham messages)

Target variable: label (0 = ham, 1 = spam)

Features: message text


🔹 Approach & Pipeline

1️⃣ Data Preprocessing:

Clean text (lowercase, remove punctuation, stopwords)
Convert text → numerical features using TF-IDF

2️⃣ Train-Test Split:
Split data into training and testing sets
Ensures unbiased model evaluation

3️⃣ Model Selection:
Used Logistic Regression because:
Outputs probabilities

Interpretable
Efficient for linear text classification

4️⃣ Probability Prediction:
Used predict_proba() to obtain spam probability for each message
Enabled decision-making beyond fixed threshold (0.5)

5️⃣ Threshold Optimization:
Evaluated model performance across multiple thresholds
Observed precision-recall trade-off

Selected threshold based on business need:
High recall → catch more spam
High precision → avoid false alarms

6️⃣ Model Evaluation:
Used:

Precision
Recall
F1-score
Confusion Matrix

7️⃣ ROC Curve & AUC:

ROC curve evaluates model across all thresholds
AUC measures class-separation capability
Demonstrates ranking quality of the model

🔹 Business Interpretation

In spam or fraud detection, missing a positive case is costly

Lowering threshold increases recall but may reduce precision

Threshold should be chosen based on business risk, not accuracy


📌 Author

Saiyed Faiez Husnain
Electrical Engineering, IIT Indore

