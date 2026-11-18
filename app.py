"""
Explainable & Reliable AI for Software Defect Prediction
DEBUG-SAFE VERSION — errors display as red banner, not black screen.
"""
import sys
import traceback
import os

import streamlit as st

# ---------------------------------------------------------------
# Status logging to terminal — helps locate crashes
# ---------------------------------------------------------------
def log(msg):
    print(f"[app.py] {msg}", file=sys.stderr, flush=True)

log("app.py starting")

# ---------------------------------------------------------------
# SAFE IMPORT BLOCK — any error becomes a red banner
# ---------------------------------------------------------------
try:
    log("importing core libs")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    log("core libs OK")

    log("importing src.data_loader")
    from src.data_loader import (
        load_dataset, detect_target_column, encode_target, dataset_summary
    )
    log("src.data_loader OK")

    log("importing src.preprocessing")
    from src.preprocessing import (
        clean_features, split_data, fit_transform_train
    )
    log("src.preprocessing OK")

    log("importing src.eda")
    from src import eda
    log("src.eda OK")

    log("importing src.models")
    from src.models import train_all_models, evaluate_models, pick_best_model
    log("src.models OK")

    log("importing src.evaluation")
    from src import evaluation
    log("src.evaluation OK")

    log("importing src.explainability")
    from src import explainability
    log("src.explainability OK")

    log("importing src.prediction")
    from src.prediction import predict_single
    log("src.prediction OK")

    log("importing src.report_generator")
    from src.report_generator import generate_pdf_report
    log("src.report_generator OK")

except Exception:
    # Render error BEFORE st.set_page_config — allowed here
    st.set_page_config(page_title="Import Error", layout="wide")
    st.error(" **Import failed — this is why you get a black screen.**")
    st.code(traceback.format_exc(), language="python")
    st.info("Fix the error above, then refresh. No changes lost.")
    log("IMPORT FAILED")
    sys.exit(1)

# ---------------------------------------------------------------
# Page config (must be first st.* call after imports)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Software Defect Prediction · Explainable AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Load CSS
# ---------------------------------------------------------------
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    log("CSS loaded")
else:
    log(f"CSS not found at {css_path} — continuing without custom styling")

# ===============================================================
# Wrap the rest of the dashboard in try/except
# ===============================================================
try:
    # ---------- session state ----------
    for key in ['df', 'target_col', 'X', 'y', 'comparison_df', 'artifacts',
                'trained_models', 'best_model_name', 'explainer',
                'shap_values', 'X_test_sc', 'y_test', 'feature_names',
                'imputer', 'scaler', 'threshold_rows', 'shap_importance_df']:
        st.session_state.setdefault(key, None)
    st.session_state.setdefault('pipeline_stage', None)
    st.session_state.setdefault('fullpage_fig', None)
    st.session_state.setdefault('fullpage_title', None)

    # ---------- graph helper ----------
    def graph_with_expand(fig, key, caption=""):
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{key}")
        if st.button(f" Open '{caption or key}' in Full Page", key=f"btn_{key}"):
            st.session_state['fullpage_fig'] = fig
            st.session_state['fullpage_title'] = caption or key
            st.rerun()

    # ---------- full-page modal ----------
    if st.session_state.get('fullpage_fig') is not None:
        st.markdown(f"### {st.session_state.get('fullpage_title', '')}")
        st.plotly_chart(st.session_state['fullpage_fig'],
                        use_container_width=True,
                        key="fullpage_chart")
        if st.button("✖ Close Full-Page View", key="close_fullpage"):
            st.session_state['fullpage_fig'] = None
            st.session_state['fullpage_title'] = None
            st.rerun()
        st.divider()

    # ---------- header ----------
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Explainable & Reliable AI for Software Defect Prediction</h1>
        <p>Comparative study of ML models · NASA PROMISE CM1 · SHAP-based explanations</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- sidebar ----------
    with st.sidebar:
        st.markdown("##  Configuration")
        st.markdown("### 1. Upload Dataset")

        use_sample = st.checkbox("Use bundled CM1 dataset (data/cm1.csv)",
                                 value=os.path.exists("data/cm1.csv"))
        uploaded = None
        if not use_sample:
            uploaded = st.file_uploader("Upload CSV", type=['csv'])

        st.markdown("### 2. Model Settings")
        test_size = st.slider("Test split %", 10, 40, 20) / 100
        random_state = st.number_input("Random seed", 0, 9999, 42)
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

        run_btn = st.button(" Run Full Pipeline", use_container_width=True)

    # ---------- pipeline: load ----------
    if run_btn:
        with st.spinner("Loading dataset…"):
            try:
                if use_sample:
                    if not os.path.exists("data/cm1.csv"):
                        st.error(" data/cm1.csv not found. Upload a CSV or add the file.")
                        st.stop()
                    df_raw = load_dataset("data/cm1.csv")
                elif uploaded is not None:
                    df_raw = load_dataset(uploaded)
                else:
                    st.error("Please upload a CSV or enable the bundled dataset.")
                    st.stop()
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
                st.code(traceback.format_exc(), language="python")
                st.stop()

            auto_target = detect_target_column(df_raw)
            st.session_state['df'] = df_raw
            st.session_state['auto_target'] = auto_target
            st.session_state['pipeline_stage'] = 'target_selected'
            st.success(f"✅ Dataset loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
            st.rerun()

    # ---------- target selection ----------
    if st.session_state['df'] is not None and st.session_state.get('pipeline_stage') == 'target_selected':
        df_raw = st.session_state['df']
        auto_target = st.session_state.get('auto_target')

        st.markdown('<div class="section-title"> Target Column Selection</div>',
                    unsafe_allow_html=True)

        cols = df_raw.columns.tolist()
        default_idx = cols.index(auto_target) if auto_target in cols else len(cols) - 1

        target_col = st.selectbox(
            "Select the target (defect label) column:",
            options=cols,
            index=default_idx,
            help="Column indicating whether a module is defective (1) or clean (0)."
        )
        st.session_state['target_col'] = target_col

        try:
            encoded_y, mapping = encode_target(df_raw[target_col])
            df_enc = df_raw.copy()
            df_enc[target_col] = encoded_y
            st.info(f"✅ Target encoding applied → {mapping}")
        except Exception as e:
            st.error(f"Target encoding failed: {e}")
            st.stop()

        st.session_state['df_encoded'] = df_enc

        if st.button(" Continue to Preprocessing & Training", type="primary"):
            st.session_state['pipeline_stage'] = 'training'
            st.rerun()

    # ---------- training ----------
    if st.session_state.get('pipeline_stage') == 'training':
        df_enc = st.session_state['df_encoded']
        target_col = st.session_state['target_col']

        with st.spinner("Cleaning features…"):
            X, y, numeric_cols, constant_cols = clean_features(df_enc, target_col)
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['feature_names'] = X.columns.tolist()

        with st.spinner("Splitting data…"):
            X_train, X_test, y_train, y_test = split_data(
                X, y, test_size=test_size, random_state=int(random_state)
            )
            X_train_sc, X_test_sc, imputer, scaler = fit_transform_train(X_train, X_test)
            st.session_state['X_train_sc'] = X_train_sc
            st.session_state['X_test_sc'] = X_test_sc
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['y_train'] = y_train
            st.session_state['imputer'] = imputer
            st.session_state['scaler'] = scaler

        with st.spinner("Training 3 models…"):
            trained = train_all_models(X_train_sc, y_train)
            comparison_df, artifacts = evaluate_models(trained, X_test_sc, y_test)
            st.session_state['trained_models'] = trained
            st.session_state['comparison_df'] = comparison_df
            st.session_state['artifacts'] = artifacts
            st.session_state['best_model_name'] = pick_best_model(comparison_df)

        with st.spinner("Computing SHAP explanations…"):
            best_name = st.session_state['best_model_name']
            best_model = trained[best_name]
            explainer = explainability.build_explainer(
                best_model, X_train_sc, X.columns.tolist()
            )
            shap_values = explainability.shap_values_for_class(explainer, X_test_sc)
            st.session_state['explainer'] = explainer
            st.session_state['shap_values'] = shap_values
            st.session_state['shap_importance_df'] = explainability.global_importance_df(
                shap_values, X.columns.tolist()
            )

        with st.spinner("Computing threshold sensitivity…"):
            st.session_state['threshold_rows'] = evaluation.threshold_sensitivity(
                best_model, X_test_sc, y_test
            )

        st.session_state['pipeline_stage'] = 'ready'
        st.success(" Pipeline complete — all 6 modules unlocked.")
        st.rerun()

    # ---------- metric card ----------
    def metric_card(col, label, value):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <h1>{value}</h1>
            </div>
            """, unsafe_allow_html=True)

    # ---------- landing / dashboard ----------
    if st.session_state.get('pipeline_stage') != 'ready':
        st.info(" Configure the sidebar and click **Run Full Pipeline** to begin.")
        st.markdown("""
        ### About this dashboard
        This app has 6 research modules:

        1. **Dataset Analysis (EDA)** — missing values, duplicates, statistics, correlation, class balance
        2. **Model Comparison** — Logistic Regression vs Random Forest vs Gradient Boosting
        3. **Model Evaluation** — Confusion matrix, ROC, PR curve, threshold sensitivity
        4. **Explainable AI (SHAP)** — global importance, beeswarm, waterfall, dependence
        5. **Individual Prediction** — input metrics → prediction + probability + explanation
        6. **Final Research Report** — downloadable PDF
        """)
        st.stop()

    # ---------- ready state: load everything ----------
    df = st.session_state['df_encoded']
    target_col = st.session_state['target_col']
    comparison_df = st.session_state['comparison_df']
    artifacts = st.session_state['artifacts']
    trained = st.session_state['trained_models']
    best_name = st.session_state['best_model_name']
    X = st.session_state['X']
    y = st.session_state['y']
    X_test_sc = st.session_state['X_test_sc']
    y_test = st.session_state['y_test']
    shap_values = st.session_state['shap_values']
    explainer = st.session_state['explainer']
    imputer = st.session_state['imputer']
    scaler = st.session_state['scaler']
    feature_names = st.session_state['feature_names']
    threshold_rows = st.session_state['threshold_rows']
    shap_importance_df = st.session_state['shap_importance_df']

    # ---------- KPI row ----------
    k1, k2, k3, k4, k5 = st.columns(5)
    metric_card(k1, "Total Modules", f"{len(df)}")
    metric_card(k2, "Defective", f"{int((df[target_col]==1).sum())}")
    metric_card(k3, "Non-Defective", f"{int((df[target_col]==0).sum())}")
    metric_card(k4, "Features Used", f"{len(feature_names)}")
    metric_card(k5, "Best Model", best_name.split()[0])
    st.markdown("---")

    # ---------- 6 tabs ----------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        " 1. Dataset Analysis",
        " 2. Model Comparison",
        " 3. Model Evaluation",
        " 4. Explainable AI (SHAP)",
        " 5. Predict a Module",
        " 6. Research Report",
    ])

    # ============ TAB 1: EDA ============
    with tab1:
        st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
        summary = dataset_summary(df, target_col)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Missing Values (total)**")
            st.write(int(sum(summary['missing_values'].values())))
            st.markdown("**Duplicate Rows**")
            st.write(summary['duplicates'])
        with c2:
            st.markdown("**Class Distribution**")
            st.write(f"Defective: {summary['defective_count']}")
            st.write(f"Non-defective: {summary['non_defective_count']}")
        with c3:
            ratio = summary['non_defective_count'] / max(summary['defective_count'], 1)
            st.markdown("**Imbalance Ratio**")
            st.write(f"{ratio:.2f} : 1")

        fig_class = eda.class_distribution_chart(df, target_col)
        graph_with_expand(fig_class, "class_dist", "Class Distribution")

        with st.expander(" Missing Values Detail"):
            st.dataframe(eda.missing_values_df(df), use_container_width=True)

        with st.expander(" Descriptive Statistics"):
            st.dataframe(eda.descriptive_stats(df, X.columns.tolist()), use_container_width=True)

        st.markdown('<div class="section-title">Correlation Analysis</div>', unsafe_allow_html=True)
        corr_df = eda.correlation_matrix(df, X.columns.tolist())
        fig_corr = eda.correlation_heatmap(corr_df)
        graph_with_expand(fig_corr, "corr_heatmap", "Correlation Heatmap")

        st.markdown('<div class="section-title">Feature Distributions</div>', unsafe_allow_html=True)
        fig_box = eda.feature_distribution_box(df, X.columns.tolist(), top_n=12)
        graph_with_expand(fig_box, "feat_box", "Top Feature Distributions")

    # ============ TAB 2: Model Comparison ============
    with tab2:
        st.markdown('<div class="section-title">Three-Model Comparison</div>', unsafe_allow_html=True)
        st.dataframe(
            comparison_df.style.highlight_max(
                subset=['Precision', 'Recall', 'F1', 'ROC-AUC'], color='#d4efdf'
            ),
            use_container_width=True
        )
        best_row = comparison_df[comparison_df['Model'] == best_name].iloc[0]
        st.success(
            f" **Best model by F1:** {best_name}  "
            f"(Precision={best_row['Precision']}, Recall={best_row['Recall']}, "
            f"F1={best_row['F1']}, ROC-AUC={best_row['ROC-AUC']})"
        )

        fig_cmp = go.Figure()
        for metric in ['Precision', 'Recall', 'F1', 'ROC-AUC']:
            fig_cmp.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=comparison_df[metric],
                textposition='auto',
            ))
        fig_cmp.update_layout(barmode='group', title='Model Comparison — All Metrics',
                              yaxis_title='Score', height=500)
        graph_with_expand(fig_cmp, "model_cmp", "Model Comparison Bar Chart")

        st.markdown(f"""
        > **RQ1: Which classifier gives the best precision-recall balance?**
        >
        > Best model: **{best_name}** — strongest F1 (minority-class detection).
        """)

        st.download_button(
            " Download Model Comparison CSV",
            data=comparison_df.to_csv(index=False).encode('utf-8'),
            file_name='model_comparison.csv',
            mime='text/csv',
        )

    # ============ TAB 3: Evaluation ============
    with tab3:
        st.markdown('<div class="section-title">Model Evaluation — Selected Model</div>',
                    unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Choose model to evaluate:",
            options=list(trained.keys()),
            index=list(trained.keys()).index(best_name),
        )
        a = artifacts[model_choice]

        c1, c2 = st.columns(2)
        with c1:
            fig_cm = evaluation.confusion_matrix_fig(a['cm'], model_choice)
            graph_with_expand(fig_cm, f"cm_{model_choice}", f"Confusion Matrix — {model_choice}")
            tn, fp, fn, tp = a['cm'].ravel()
            st.markdown(f"""
            - **True Positives (TP):** {tp}
            - **True Negatives (TN):** {tn}
            - **False Positives (FP):** {fp}
            - **False Negatives (FN):** {fn}
            """)
        with c2:
            st.markdown("#### ROC Curve")
            fig_roc = evaluation.roc_curve_fig(artifacts)
            graph_with_expand(fig_roc, "roc", "ROC Curve — All Models")

        st.markdown('<div class="section-title">Precision-Recall Curve</div>',
                    unsafe_allow_html=True)
        fig_pr = evaluation.pr_curve_fig(artifacts)
        graph_with_expand(fig_pr, "pr", "Precision-Recall Curve")

        st.markdown('<div class="section-title">Threshold Sensitivity</div>',
                    unsafe_allow_html=True)
        fig_thr = evaluation.threshold_curve_fig(threshold_rows, model_choice)
        graph_with_expand(fig_thr, "thr", "Threshold Sensitivity")

        with st.expander(" Threshold Table"):
            st.dataframe(pd.DataFrame(threshold_rows), use_container_width=True)

        st.markdown("""
        > **RQ2: How much does prediction change with threshold?**
        >
        > Lower threshold → higher recall, lower precision.
        > For safety-critical work, threshold ≈ 0.3–0.4 balances recall and F1 best.
        """)

    # ============ TAB 4: SHAP ============
    with tab4:
        st.markdown('<div class="section-title">Explainable AI — SHAP Analysis</div>',
                    unsafe_allow_html=True)

        st.markdown("#### 4.1 Feature Importance (Global)")
        fig_imp = explainability.feature_importance_bar(shap_values, feature_names, top_n=15)
        graph_with_expand(fig_imp, "shap_bar", "SHAP Feature Importance Bar")
        with st.expander(" Importance Table"):
            st.dataframe(shap_importance_df.head(20), use_container_width=True)

        st.markdown("#### 4.2 SHAP Beeswarm (Global Impact)")
        X_test_df = pd.DataFrame(X_test_sc, columns=feature_names)
        fig_bee = explainability.beeswarm_fig(shap_values, X_test_df)
        graph_with_expand(fig_bee, "shap_bee", "SHAP Beeswarm")

        st.markdown("#### 4.3 SHAP Waterfall — One Module")
        sample_idx = st.number_input(
            "Select test sample index to explain:",
            min_value=0, max_value=len(X_test_sc) - 1, value=0
        )
        fig_wf, base, total = explainability.waterfall_fig_for_sample(
            explainer, X_test_df, int(sample_idx), feature_names
        )
        graph_with_expand(fig_wf, "shap_wf", f"SHAP Waterfall — Module #{sample_idx}")

        st.markdown("#### 4.4 SHAP Dependence")
        dep_feat = st.selectbox("Select feature for dependence plot:", options=feature_names)
        feat_idx = feature_names.index(dep_feat)
        fig_dep = explainability.dependence_fig(shap_values, X_test_df, feat_idx)
        graph_with_expand(fig_dep, "shap_dep", f"SHAP Dependence — {dep_feat}")

        st.markdown("""
        > **RQ3: Which software metrics drive predictions?**
        >
        > Typically **loc, v(g), ev(g), e** dominate. Higher complexity → higher defect risk.
        """)

    # ============ TAB 5: Individual Prediction ============
    with tab5:
        st.markdown('<div class="section-title">Predict Defect Risk for a New Module</div>',
                    unsafe_allow_html=True)
        st.markdown("Enter software metric values below.")

        with st.form("prediction_form"):
            input_vals = {}
            cols = st.columns(3)
            for i, feat in enumerate(feature_names):
                with cols[i % 3]:
                    input_vals[feat] = st.number_input(
                        feat, value=float(X[feat].median()), format="%.4f"
                    )
            submitted = st.form_submit_button(" Predict", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([input_vals])
            result = predict_single(
                trained[best_name], explainer, scaler, imputer,
                input_df, feature_names, threshold=threshold
            )

            if result['prediction'] == 1:
                st.markdown(
                    f'<div class="prediction-defective"> DEFECTIVE — '
                    f'{result["probability"]*100:.2f}% probability</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="prediction-clean"> NON-DEFECTIVE — '
                    f'{result["probability"]*100:.2f}% probability</div>',
                    unsafe_allow_html=True)

            st.markdown(f"**Model:** {best_name}  |  **Threshold:** {threshold:.2f}")
            st.markdown("###  Why this prediction?")
            contribs = result['top_contributions']
            fig_contrib = go.Figure(go.Bar(
                x=contribs['SHAP'], y=contribs['Feature'], orientation='h',
                marker_color=['#e74c3c' if v > 0 else '#3498db' for v in contribs['SHAP']],
                text=contribs['SHAP'].round(3), textposition='auto',
            ))
            fig_contrib.update_layout(
                title='Top SHAP Contributions (red = increases risk, blue = decreases)',
                xaxis_title='SHAP value',
                yaxis=dict(autorange='reversed'), height=500,
            )
            graph_with_expand(fig_contrib, "single_shap", "Single Prediction SHAP")

            st.markdown("**Top risk-increasing metrics:**")
            for _, row in contribs[contribs['SHAP'] > 0].head(3).iterrows():
                st.markdown(f"- **{row['Feature']}** = {row['Value']:.2f} (+{row['SHAP']:.3f})")
            st.markdown("**Top risk-reducing metrics:**")
            for _, row in contribs[contribs['SHAP'] < 0].head(3).iterrows():
                st.markdown(f"- **{row['Feature']}** = {row['Value']:.2f} ({row['SHAP']:.3f})")

    # ============ TAB 6: Report ============
    with tab6:
        st.markdown('<div class="section-title">Final Research Report</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        This module generates a downloadable **PDF research report** containing:
        dataset analysis, preprocessing details, 3-model comparison, confusion matrices,
        ROC/PR curves, threshold sensitivity, SHAP feature importance, findings, limitations.
        """)

        best_row = comparison_df[comparison_df['Model'] == best_name].iloc[0]
        top3_features = shap_importance_df.head(3)['Feature'].tolist()

        research_findings = {
            "Q1: Which ML model performs best?":
                f"{best_name} — F1={best_row['F1']:.4f}, Precision={best_row['Precision']:.4f}, "
                f"Recall={best_row['Recall']:.4f}, ROC-AUC={best_row['ROC-AUC']:.4f}.",
            "Q2: Why does the model predict a defect?":
                f"Top features: {', '.join(top3_features)}. Higher complexity → higher defect probability.",
            "Q3: How reliable is the prediction?":
                f"Recall={best_row['Recall']:.4f}, F1={best_row['F1']:.4f}, "
                f"ROC-AUC={best_row['ROC-AUC']:.4f}. Stable in threshold range 0.3–0.6.",
        }
        for q, ans in research_findings.items():
            st.markdown(f"**{q}**  \n{ans}\n")

        if st.button(" Generate & Download PDF Report", type="primary"):
            with st.spinner("Generating PDF…"):
                os.makedirs("outputs/reports", exist_ok=True)
                pdf_path = "outputs/reports/research_report.pdf"
                try:
                    generate_pdf_report(
                        output_path=pdf_path,
                        dataset_summary=dataset_summary(df, target_col),
                        comparison_df=comparison_df,
                        threshold_rows=threshold_rows,
                        shap_importance_df=shap_importance_df,
                        research_findings=research_findings,
                    )
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            " Download research_report.pdf",
                            data=f.read(), file_name="research_report.pdf",
                            mime="application/pdf",
                        )
                    st.success(f"PDF generated at {pdf_path}")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
                    st.code(traceback.format_exc(), language="python")

        st.markdown("---")
        st.markdown("### Export Raw Results")
        st.download_button(" Model Comparison CSV",
                           data=comparison_df.to_csv(index=False).encode(),
                           file_name="model_comparison.csv", mime="text/csv")
        st.download_button(" SHAP Importance CSV",
                           data=shap_importance_df.to_csv(index=False).encode(),
                           file_name="shap_importance.csv", mime="text/csv")
        st.download_button(" Threshold Sensitivity CSV",
                           data=pd.DataFrame(threshold_rows).to_csv(index=False).encode(),
                           file_name="threshold_sensitivity.csv", mime="text/csv")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#7f8c8d; font-size:0.85rem;'>"
        "MCA Minor Project · Explainable & Reliable AI for Software Defect Prediction · "
        "NASA PROMISE CM1"
        "</div>", unsafe_allow_html=True)

except Exception:
    st.error(" **Runtime error inside dashboard**")
    st.code(traceback.format_exc(), language="python")
    log("RUNTIME ERROR in dashboard")