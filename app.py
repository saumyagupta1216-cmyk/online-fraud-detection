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
    st.write("Upload a labelled transaction CSV. For your dataset, select Fraudulent (0 = legitimate, 1 = fraud). Numeric and text columns are supported.")
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
    st.warning("This is an educational estimate, not a definitive fraud finding or a connection to a bank.")
    df, source = get_data()
    if df is None:
        st.info("Upload your labelled CSV in Dataset Analysis first.")
    else:
        target = st.session_state.get("fraud_target", target_col(df))
        if not target or target not in df.columns or df[target].nunique(dropna=True) != 2:
            st.error("Choose the binary Fraudulent target in Dataset Analysis first.")
        else:
            X, y = prepare_xy(df, target)
            with st.form("transaction_form"):
                st.write("Enter transaction details. ID columns are excluded from prediction.")
                values = {}
                cols = st.columns(2)
                for i, feature in enumerate(X.columns):
                    with cols[i % 2]:
                        if pd.api.types.is_numeric_dtype(X[feature]):
                            series = pd.to_numeric(X[feature], errors="coerce").dropna()
                            default = float(series.median()) if not series.empty else 0.0
                            values[feature] = st.number_input(feature, value=default, format="%.4f", key=f"feature_{feature}")
                        else:
                            options = X[feature].dropna().astype(str).value_counts().index.tolist()
                            options = options or ["Unknown"]
                            values[feature] = st.selectbox(feature, options, key=f"feature_{feature}")
                model_name = st.selectbox("Classifier", list(build_models().keys()), index=list(build_models().keys()).index("Random Forest"), key="prediction_model")
                submitted = st.form_submit_button("Train / load model and analyse", type="primary")
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
    st.write("Compare classifiers using cross-validation, then train and save one model for transaction checks.")
    df, source = get_data()
    if df is None:
        st.info("Upload a labelled CSV in Dataset Analysis first.")
    else:
        target = st.session_state.get("fraud_target", target_col(df))
        if not target or target not in df.columns:
            target = st.selectbox("Target column", list(df.columns))
        try:
            X, y = prepare_xy(df, target)
            model_names = list(build_models().keys())
            chosen_models = st.multiselect("Models to compare", model_names, default=["Random Forest"])
            folds = st.select_slider("Cross-validation folds", options=[3, 5, 10], value=3)
            run = st.button("Run model comparison", type="primary", disabled=not chosen_models)
            if run:
                if y.value_counts().min() < folds:
                    st.error(f"Need at least {folds} examples in each class.")
                else:
                    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
                    scoring = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1 Score": "f1", "ROC-AUC": "roc_auc"}
                    models = build_models(X)
                    rows = []
                    progress = st.progress(0, text="Evaluating models…")
                    for i, name in enumerate(chosen_models):
                        start_time = time.time()
                        try:
                            scores = cross_validate(models[name], X, y, cv=cv, scoring=scoring, error_score="raise", n_jobs=None)
                            rows.append({"Model": name, **{metric: float(np.mean(scores[f"test_{metric}"])) for metric in scoring}, "Time (sec)": time.time() - start_time})
                        except Exception as e:
                            st.warning(f"{name} could not be evaluated: {e}")
                        progress.progress((i + 1) / len(chosen_models), text=f"Evaluated {i + 1} of {len(chosen_models)}")
                    if rows:
                        result = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
                        st.session_state["model_comparison"] = result
                        st.dataframe(result.style.format({c: "{:.4f}" for c in result.select_dtypes("number").columns}), use_container_width=True)
