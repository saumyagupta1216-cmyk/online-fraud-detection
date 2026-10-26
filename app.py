"""
Online Fraud Detection — Streamlit web application
Concepts adapted from the user's two uploaded experiment documents:
- Experiment 1.3: supervised learning workflow, train/test split, StandardScaler,
  Gaussian Naive Bayes, stratified 10-fold CV, confusion matrix, ROC curve,
  Accuracy, Precision, Recall, F1, ROC-AUC, Cohen's Kappa, MCC.
- Experiment 1.4: comparing classification algorithms using stratified CV.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, cohen_kappa_score, matthews_corrcoef
)

st.set_page_config(page_title="FraudShield | Online Fraud Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg, #0b1220 0%, #101b2d 100%); color:#eef4ff;}
[data-testid="stSidebar"] {background:#111c2d;}
.hero {padding: 1.4rem 1.6rem; border:1px solid #2b405e; border-radius:18px;
background:linear-gradient(120deg,#14263e,#162f3a); margin-bottom:1rem;}
.hero h1 {font-size:2.35rem; margin:0; color:#f3f7ff;}
.hero p {color:#b8c9de; font-size:1rem; margin:.45rem 0 0;}
.kpi {background:#14243a; border:1px solid #2c4260; padding:1rem; border-radius:14px;}
.kpi-label {color:#9eb3cc; font-size:.85rem;}
.kpi-value {font-size:1.55rem; font-weight:700; color:#f6fbff;}
.small-note {color:#aabbd0; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛡️ FraudShield")
st.sidebar.caption("Machine-learning fraud analysis")
page = st.sidebar.radio("Navigate", ["Overview", "Check a Transaction", "Model Lab", "Dataset Analysis", "About & Concepts"])
st.sidebar.divider()
st.sidebar.caption("Educational prototype • No real payment processing")

@st.cache_data
def load_csv(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

def build_models():
    """Return model estimators, or full preprocessing pipelines when X is supplied."""
    raw = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=120, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
    }
    if X is None:
        return raw

    # IDs usually identify a row/person rather than describe transaction risk.
    X = X.drop(columns=[c for c in ["Transaction_ID", "User_ID"] if c in X.columns], errors="ignore")
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return {
        name: Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", estimator),
        ]) for name, estimator in raw.items()
    }

def target_col(df):
    candidates = ["is_fraud", "fraud", "Class", "class", "target", "label", "isFraud", "is_fraudulent", "Fraudulent"]
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None

def score_model(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    out = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "Cohen's Kappa": cohen_kappa_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    }
    out["ROC-AUC"] = roc_auc_score(y_test, prob) if prob is not None and len(np.unique(y_test)) == 2 else np.nan
    return pred, prob, out

def prepare_xy(df, target):
    """Prepare labelled data while keeping numeric and categorical features."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    X = df.loc[valid].drop(columns=[target]).copy()
    y = y.loc[valid].astype(int)

    # Remove identifier columns so the model learns transaction patterns, not IDs.
    X = X.drop(columns=[c for c in ["Transaction_ID", "User_ID"] if c in X.columns], errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No feature columns are available after removing IDs.")
    if y.nunique() != 2:
        raise ValueError("Choose a binary target column, such as Fraudulent (0 = legitimate, 1 = fraud).")
    if y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two examples.")
    return X, y


def train_experiment(df, target, model_name, test_size=0.2):
    X, y = prepare_xy(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = build_models(X)[model_name]
    model.fit(X_train, y_train)
    pred, prob, metrics = score_model(model, X_test, y_test)
    return X, y, X_train, X_test, y_train, y_test, model, pred, prob, metrics

def save_trained_model(model, X, target, model_name):
    """Save the complete pipeline and the exact feature order used at training."""
    joblib.dump({
        "model": model,
        "features": list(X.columns),
        "target": target,
        "model_name": model_name,
    }, "fraud_model.joblib")

def get_data():
    if "fraud_df" in st.session_state:
        return st.session_state["fraud_df"], st.session_state.get("fraud_source", "Uploaded dataset")
    return None, None

st.markdown("""
<div class="hero">
  <h1>🛡️ FraudShield</h1>
  <p>Online Fraud Detection • Supervised learning, transaction risk signals, and model evaluation.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.subheader("Fraud monitoring dashboard")
    st.write("Explore a transaction dataset, train a binary classifier, and compare models using standard machine-learning metrics.")
    df, source = get_data()
    if df is None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset", "Not loaded")
        c2.metric("Model", "Not trained")
        c3.metric("Risk checks", "Demo only")
        st.info("Start by opening **Dataset Analysis** and uploading a labelled CSV. A `0/1` fraud label is recommended.")
    else:
        tcol = target_col(df)
        a, b, c, d = st.columns(4)
        a.metric("Transactions / rows", f"{len(df):,}")
        b.metric("Available features", str(max(0, len(df.columns)-1)))
        bcol = df[tcol] if tcol else pd.Series(dtype=float)
        fraud_count = int((pd.to_numeric(bcol, errors="coerce") == 1).sum()) if tcol else 0
        c.metric("Rows labelled fraud (1)", f"{fraud_count:,}" if tcol else "Target not detected")
        d.metric("Dataset source", "Uploaded CSV")
        st.caption(f"Loaded: {source}")
        if tcol:
            rate = (pd.to_numeric(df[tcol], errors="coerce") == 1).mean() * 100
            st.progress(float(np.clip(rate/100, 0, 1)), text=f"Positive-label share: {rate:.2f}% (assuming 1 = fraud)")
        st.dataframe(df.head(10), use_container_width=True)
    st.markdown("### Workflow")
    steps = st.columns(5)
    for col, num, title in zip(steps, ["01","02","03","04","05"], ["Upload data","Preprocess","Train","Evaluate","Compare"]):
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-label">STEP {num}</div><div class="kpi-value">{title}</div></div>', unsafe_allow_html=True)

elif page == "Dataset Analysis":
    st.subheader("Dataset workspace")
    st.write("Upload a labelled transaction CSV. The target should indicate whether a transaction is fraudulent (1) or legitimate (0).")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded:
        try:
            df = load_csv(uploaded.getvalue())
            st.session_state["fraud_df"] = df
            st.session_state["fraud_source"] = uploaded.name
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    df, source = get_data()
    if df is not None:
        st.success(f"{source} loaded successfully.")
        left, right = st.columns(2)
        with left:
            st.markdown("**Preview**")
            st.dataframe(df.head(15), use_container_width=True)
        with right:
            st.markdown("**Dataset profile**")
            st.write(f"Rows: **{df.shape[0]:,}** · Columns: **{df.shape[1]}**")
            st.write("Missing values by column")
            missing = df.isna().sum().rename("Missing values").to_frame()
            st.dataframe(missing, use_container_width=True)
        st.markdown("**Numeric summary**")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        guessed = target_col(df)
        options = list(df.columns)
        default_idx = options.index(guessed) if guessed in options else 0
        chosen = st.selectbox("Target / label column", options, index=default_idx, help="Choose the column that labels fraud vs legitimate transactions.")
        st.session_state["fraud_target"] = chosen
        if df[chosen].nunique(dropna=True) == 2:
            counts = df[chosen].value_counts(dropna=False).rename_axis("Class").reset_index(name="Count")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(data=counts, x="Class", y="Count", ax=ax)
            ax.set_title("Class distribution")
            st.pyplot(fig)
        else:
            st.warning("Selected target does not have exactly two classes. Choose a binary fraud label for this prototype.")
    else:
        st.markdown("#### Expected CSV structure")
        st.code("amount, transaction_hour, account_age_days, failed_attempts, is_fraud\n125.50, 14, 420, 0, 0\n980.00, 2, 15, 3, 1", language="csv")
        st.caption("These rows are illustrative examples only, not real transactions or a validated dataset.")

elif page == "Check a Transaction":
    st.subheader("Transaction risk check")
    st.warning("A risk estimate is only meaningful after training on a suitable labelled fraud dataset. This demo does not connect to banks or process payments.")
    df, source = get_data()
    if df is None:
        st.info("Upload a labelled CSV in Dataset Analysis first.")
    else:
        target = st.session_state.get("fraud_target", target_col(df))
        if not target or target not in df.columns or df[target].nunique(dropna=True) != 2:
            st.error("Choose a binary fraud target in Dataset Analysis first.")
        else:
            features = df.drop(columns=[target]).select_dtypes(include=[np.number]).columns.tolist()
            if not features:
                st.error("No numeric features are available.")
            else:
                X, y = prepare_xy(df, target)
                with st.form("transaction_form"):
                    st.write("Enter values for the numeric features in your dataset:")
                    values = {}
                    cols = st.columns(2)
                    for i, feature in enumerate(features):
                      with cols[i % 2]:
                        if pd.api.types.is_numeric_dtype(X[feature]):
                           series = pd.to_numeric(df[feature], errors="coerce").dropna()
                           default = float(series.median()) if not series.empty else 0.0
                           values[feature] = st.number_input(feature, value=default, format="%.4f", key=f"feature_{feature}")
                        else:
                            options = X[feature].dropna().astype(str).value_counts().index.tolist()
                            options = options or ["Unknown"]
                            values[feature] = st.selectbox(feature, options, key=f"feature_{feature}")
                 model_name = st.selectbox("Classifier", list(build_models().keys()), key="prediction_model")
                 submitted = st.form_submit_button("Analyse transaction", type="primary")
            if submitted:
                 try:
                    saved = Path("fraud_model.joblib")
                    saved_bundle = None
                    if saved.exists():
                        try:
                            saved_bundle = joblib.load(saved)
                        except Exception:
                            saved_bundle = None
                    if saved_bundle and saved_bundle.get("target") == target and saved_bundle.get("model_name") == model_name and saved_bundle.get("features") == list(X.columns):
                        model = saved_bundle["model"]
                        st.info("Loaded the previously trained model. It was trained on the full uploaded dataset, so separate test metrics are not shown here.")
                        metrics = None
                    else:
                        X_all, y_all, X_train, X_test, y_train, y_test, model, pred, prob, metrics = train_experiment(df, target, model_name)
                        save_trained_model(model, X_all, target, model_name)
                        st.success("Model trained and saved as fraud_model.joblib.")
                    row = pd.DataFrame([values], columns=X.columns)
                    label = int(model.predict(row)[0])
                    probability = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else None
                    if label == 1:
                        st.error("Model classification: FRAUD-LIKE")
                    else:
                        st.success("Model classification: LEGITIMATE-LIKE")
                    if probability is not None:
                        st.metric("Estimated fraud probability", f"{probability * 100:.2f}%")
                    st.caption("Confirm that the dataset's positive class means fraud. Predictions are not guaranteed to be correct.")
                    if metrics is not None:
                        st.markdown("**Held-out test performance**")
                        st.json({k: round(float(v), 4) for k, v in metrics.items() if pd.notna(v)})
                except Exception as e:
                    st.error(f"Could not train or score model: {e}")
   

elif page == "Model Lab":
    st.subheader("Classification model laboratory")
    st.write("Compare several supervised classifiers with stratified 10-fold cross-validation, inspired by Experiment 1.4.")
    df, source = get_data()
    if df is None:
        st.info("Upload a labelled CSV in Dataset Analysis first.")
    else:
        target = st.session_state.get("fraud_target", target_col(df))
        if not target or target not in df.columns:
            target = st.selectbox("Target column", list(df.columns))
        try:
          X, y = prepare_xy(df, target)
          models = build_models()
      ]   chosen_models = st.multiselect("Models to compare", list(models.keys()), default=["Gaussian Naive Bayes", "Logistic Regression", "Decision Tree", "Random Forest"])
          folds = st.select_slider("Cross-validation folds", options=[3, 5, 10], value=5)
          run = st.button("Run model comparison", type="primary", disabled=not chosen_models)
          if run:
                  if y.value_counts().min() < folds:
                    st.error(f"Need at least {folds} examples in each class.")
                  else:
                    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
                    scoring = {"Accuracy":"accuracy", "Precision":"precision", "Recall":"recall", "F1 Score":"f1", "ROC-AUC":"roc_auc"}
                    rows = []
                    progress = st.progress(0, text="Evaluating models…")
                    for i, name in enumerate(chosen_models):
                      start = time.time()
                      try:
                          scores = cross_validate(models[name], X, y, cv=cv, scoring=scoring, error_score="raise", n_jobs=None)
                          rows.append({"Model":name, **{metric:float(np.mean(scores[f"test_{metric}"])) for metric in scoring}, "Time (sec)":time.time()-start})
                      except Exception as e:
                          st.warning(f"{name} could not be evaluated: {e}")
                      progress.progress((i+1)/len(chosen_models), text=f"Evaluated {i+1} of {len(chosen_models)}")
                 if rows:
                     result = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
                     st.session_state["model_comparison"] = result
                     st.dataframe(result.style.format({c:"{:.4f}" for c in result.select_dtypes("number").columns}), use_container_width=True)
                     fig, ax = plt.subplots(figsize=(10, 4))
                     sns.barplot(data=result, x="Model", y="F1 Score", ax=ax)
                     ax.set_ylim(0, 1)
                     ax.tick_params(axis="x", rotation=35)
                     ax.set_title("Cross-validation F1 comparison")
                     st.pyplot(fig)
                     st.download_button("Download comparison CSV", result.to_csv(index=False), "model_comparison.csv", "text/csv")
          st.markdown("### Train and save a model")
            chosen_to_save = st.selectbox("Model to train and save", model_names, index=model_names.index("Random Forest"))
            if st.button("Train and save selected model", type="primary", key="save_model_button"):
                with st.spinner("Training model and preparing the saved file…"):
                    X_train, y_train = X, y
                    saved_model = build_models(X)[chosen_to_save]
                    saved_model.fit(X_train, y_train)
                    save_trained_model(saved_model, X, target, chosen_to_save)
                st.success("Training complete! Saved as fraud_model.joblib.")
                with open("fraud_model.joblib", "rb") as model_file:
                    st.download_button("Download trained model", model_file, file_name="fraud_model.joblib", mime="application/octet-stream")
        except Exception as e:
            st.error(f"Could not prepare dataset: {e}")
    st.markdown("#### Included algorithms")
    st.caption("Gaussian Naive Bayes, Logistic Regression, Decision Tree, KNN, SVM, Random Forest, Extra Trees, Gradient Boosting, and AdaBoost.")

elif page == "About & Concepts":
    st.subheader("Concepts used from your experiment documents")
    st.markdown("""
    **Experiment 1.3 — supervised learning and classifier evaluation**
    - **Features (X) and labels (y):** transaction attributes are model inputs; the fraud label is the target.
    - **Train/test split:** default holdout is 80% training and 20% testing, stratified by class.
    - **Preprocessing:** missing numeric values are median-imputed; numerical features are standardized with `StandardScaler` inside a pipeline.
    - **Gaussian Naive Bayes:** included as a classifier from the experiment.
    - **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, Cohen's Kappa, and Matthews Correlation Coefficient (MCC).
    - **Cross-validation:** stratified folds help assess performance across class-balanced folds.

    **Experiment 1.4 — comparing classification algorithms**
    - Compare multiple classifiers under the same stratified cross-validation scheme.
    - View mean Accuracy, Precision, Recall, F1-score, ROC-AUC, and execution time.
    - Algorithms include Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Extra Trees, Gradient Boosting, and AdaBoost.

    **Important scope note:** the uploaded experiments use a heart-disease dataset (`heart.csv`) as their example, not a fraud dataset. This app adapts their classification workflow to fraud detection, but real fraud performance can only be measured with a suitable labelled transaction dataset.
    """)
    st.markdown("#### How to read the metrics")
    st.markdown("""
    - **Precision:** of transactions flagged as fraud, the share that are actually fraud.
    - **Recall:** of fraudulent transactions, the share detected by the model.
    - **F1-score:** balance between precision and recall.
    - **ROC-AUC:** how well the model ranks positive examples above negative ones.
    - **Confusion matrix:** counts of true/false positives and true/false negatives.
    - **MCC / Kappa:** agreement measures that can be useful when classes are imbalanced.
    """)
    st.info("Educational prototype only. Do not use its output to approve, decline, or accuse in real financial transactions without proper validation, monitoring, privacy controls, and expert review.")

st.sidebar.divider()
st.sidebar.caption("FraudShield • Supervised ML demo")
