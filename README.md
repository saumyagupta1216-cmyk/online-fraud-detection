# FraudShield — Online Fraud Detection Website

A Python-first Streamlit website that adapts the supervised-classification concepts in the two supplied experiment documents to an educational online-fraud-detection workflow.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the site:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL Streamlit prints (usually `http://localhost:8501`).

## How to use

1. Select **Dataset Analysis** and upload a labelled CSV.
2. Choose the target column that marks fraud vs legitimate transactions. The app tries common names such as `is_fraud`, `fraud`, `Class`, `target`, and `label`.
3. Select **Model Lab** to compare classifiers with stratified cross-validation.
4. Select **Check a Transaction** to enter numeric feature values and receive a model classification.

## CSV expectations

- One row per transaction.
- Numeric feature columns such as amount, transaction hour, account age, failed attempts, etc.
- One binary label column, ideally `0 = legitimate`, `1 = fraud`.
- Remove identifiers that should not be used as predictive features. Do not upload personal or payment-card information.

The example rows shown in the app are illustrative only. No sample dataset is bundled, and no model is trained until you upload data.

## Concepts mapped from the supplied documents

### Experiment 1.3
- Supervised learning; features X and target y
- 80/20 stratified train/test split
- Numeric imputation and `StandardScaler`
- Gaussian Naive Bayes classifier
- Stratified cross-validation
- Accuracy, precision, recall, F1, ROC-AUC, Cohen's Kappa, MCC
- Dataset preview, descriptive summary, class distribution

### Experiment 1.4
- Multiple classifier comparison
- Stratified K-fold cross-validation
- Mean classification metrics and execution time
- Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Extra Trees, Gradient Boosting, AdaBoost

## Important limitation

The uploaded experiments demonstrate the workflow using a heart-disease dataset, not a fraud dataset. This website reuses the methodology, not the medical data or its model results. Use a suitable labelled fraud dataset to train and validate a fraud model. This is an educational prototype, not a production fraud detection service.
