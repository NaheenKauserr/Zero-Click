"""ml_engine.py — Auto ML: clustering, regression, classification, feature importance."""
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import utils

@st.cache_data(show_spinner=False)
def auto_ml(df: pd.DataFrame, column_types: dict) -> list:
    results = []
    if df is None or len(df) < 10:
        return results

    num_cols  = [c for c in column_types.get("numeric", []) if not any(k in c.lower() for k in ["id", "index", "code"])]
    cat_cols  = column_types.get("categorical", [])
    bool_cols = column_types.get("boolean", [])

    ml_df = df.dropna().copy()
    if len(ml_df) > 5000:
        ml_df = ml_df.sample(5000, random_state=42)
    if len(ml_df) < 10:
        return results

    # 1. K-Means Clustering
    if len(num_cols) >= 3:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            feats = num_cols[:3]
            X     = ml_df[feats]
            Xs    = (X - X.mean()) / (X.std() + 1e-8)

            km     = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = km.fit_predict(Xs)
            ml_df["Cluster"] = [str(c) for c in labels]
            score  = silhouette_score(Xs, labels)

            fig = px.scatter(ml_df, x=feats[0], y=feats[1], color="Cluster",
                             color_discrete_sequence=["#a78bfa", "#4ade80", "#f87171"],
                             title=f"K-Means Clustering · {feats[0]} vs {feats[1]}")
            results.append({
                "type": "Clustering", "title": "K-Means Clustering (k=3)",
                "metric_name": "Silhouette Score", "metric_value": f"{score:.3f}",
                "figure": utils.apply_genesis_theme(fig), "extra": f"Features: {', '.join(feats)}",
            })
        except Exception as e:
            print(f"Clustering: {e}")

    # 2. Linear Regression
    if len(num_cols) >= 2:
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score

            target = num_cols[0]
            feats  = num_cols[1:4]
            X, y   = ml_df[feats], ml_df[target]
            model  = LinearRegression()
            model.fit(X, y)
            preds  = model.predict(X)
            r2     = r2_score(y, preds)

            plot_df = pd.DataFrame({"Actual": y, "Predicted": preds})
            fig = px.scatter(plot_df, x="Actual", y="Predicted",
                             title=f"Actual vs Predicted · {target}",
                             color_discrete_sequence=["#a78bfa"])
            fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                          line=dict(color="#4ade80", width=1.5, dash="dot"))

            results.append({
                "type": "Regression", "title": f"Linear Regression → {target}",
                "metric_name": "R² Score", "metric_value": f"{r2:.3f}",
                "figure": utils.apply_genesis_theme(fig), "extra": f"Features: {', '.join(feats)}",
            })
        except Exception as e:
            print(f"Regression: {e}")

    # 3. Classification (binary target)
    bin_target = None
    if bool_cols:
        bin_target = bool_cols[0]
    else:
        for col in cat_cols:
            if ml_df[col].nunique() == 2:
                bin_target = col
                break

    if bin_target and len(num_cols) >= 1:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, confusion_matrix

            feats = num_cols[:3]
            X, y  = ml_df[feats], ml_df[bin_target]
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            preds = model.predict(X)
            acc   = accuracy_score(y, preds)
            cm    = confusion_matrix(y, preds)

            fig = px.imshow(cm, text_auto=True, title=f"Confusion Matrix · {bin_target}",
                            color_continuous_scale=[[0, "rgba(30,41,59,0.5)"], [1, "#a78bfa"]])
            results.append({
                "type": "Classification", "title": f"Logistic Regression → {bin_target}",
                "metric_name": "Accuracy", "metric_value": f"{acc:.2%}",
                "figure": utils.apply_genesis_theme(fig), "extra": f"Features: {', '.join(feats)}",
            })
        except Exception as e:
            print(f"Classification: {e}")

    # 4. Feature Importance (Random Forest)
    if len(num_cols) >= 3:
        try:
            from sklearn.ensemble import RandomForestRegressor

            target = num_cols[0]
            feats  = num_cols[1:6]
            X, y   = ml_df[feats], ml_df[target]
            rf     = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)

            imp_df = pd.DataFrame({"Feature": feats, "Importance": rf.feature_importances_})\
                       .sort_values("Importance", ascending=True)
            top_driver = imp_df.iloc[-1]["Feature"]

            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title=f"Feature Importance → {target}",
                         color_discrete_sequence=["#a78bfa"])
            results.append({
                "type": "Feature Importance", "title": f"Key Drivers → {target}",
                "metric_name": "Top Driver", "metric_value": top_driver,
                "figure": utils.apply_genesis_theme(fig), "extra": f"{len(feats)} features analyzed",
            })
        except Exception as e:
            print(f"Feature importance: {e}")

    return results
