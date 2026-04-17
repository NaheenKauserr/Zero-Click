"""forecasting.py — Time-series forecasting: Prophet preferred, H-W fallback."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import utils

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

@st.cache_data(show_spinner=False)
def auto_forecast(df: pd.DataFrame, column_types: dict):
    if df is None or df.empty:
        return None

    date_cols = column_types.get("datetime", [])
    num_cols  = [c for c in column_types.get("numeric", [])
                 if not any(k in c.lower() for k in ["id", "index", "code"])]

    if not date_cols or not num_cols:
        return None

    date_col   = date_cols[0]
    target_col = num_cols[0]
    for col in num_cols:
        if any(w in col.lower() for w in ["sales", "revenue", "amount", "value", "total"]):
            target_col = col
            break

    pdf = df[[date_col, target_col]].dropna().copy()
    pdf[date_col] = pd.to_datetime(pdf[date_col], errors="coerce")
    pdf = pdf.dropna(subset=[date_col])
    pdf = pdf.rename(columns={date_col: "ds", target_col: "y"})
    pdf = pdf.groupby("ds")["y"].sum().reset_index().sort_values("ds")

    if HAS_PROPHET and len(pdf) >= 12:
        try:
            m       = Prophet(daily_seasonality=False, yearly_seasonality=True)
            m.fit(pdf)
            periods = max(30, int(len(pdf) * 0.2))
            future  = m.make_future_dataframe(periods=periods, freq="D")
            fc      = m.predict(future)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pdf["ds"], y=pdf["y"], mode="lines",
                                     name="Historical", line=dict(color="#a78bfa", width=2)))
            fc_only = fc[fc["ds"] > pdf["ds"].max()]
            fig.add_trace(go.Scatter(x=fc_only["ds"], y=fc_only["yhat"], mode="lines",
                                     name="Forecast", line=dict(color="#fb923c", width=2.5, dash="dash")))
            fig.add_trace(go.Scatter(
                x=list(fc_only["ds"]) + list(fc_only["ds"])[::-1],
                y=list(fc_only["yhat_upper"]) + list(fc_only["yhat_lower"])[::-1],
                fill="toself", fillcolor="rgba(251,146,60,0.1)",
                line=dict(color="rgba(0,0,0,0)"), name="95% Confidence",
            ))
            fig.update_layout(title=f"Advanced Forecast · {target_col}")
            return {
                "figure": utils.apply_genesis_theme(fig),
                "forecast_df": fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12),
                "model": "Prophet",
            }
        except Exception as e:
            print(f"Prophet failed: {e}")

    # Holt-Winters fallback
    if len(pdf) >= 6:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            ts = pdf.set_index("ds")["y"]
            model = ExponentialSmoothing(ts, trend="add", seasonal=None)
            fitted = model.fit()
            fc = fitted.forecast(6)
            fc_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=6, freq="ME")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines",
                                     name="Historical", line=dict(color="#a78bfa", width=2)))
            fig.add_trace(go.Scatter(x=fc_dates, y=fc.values, mode="lines+markers",
                                     name="Forecast (HW)", line=dict(color="#fb923c", width=2, dash="dot"),
                                     marker=dict(size=6, symbol="diamond")))
            fig.update_layout(title=f"Holt-Winters Forecast · {target_col}")
            return {
                "figure": utils.apply_genesis_theme(fig),
                "forecast_df": pd.DataFrame({"ds": fc_dates, "yhat": fc.values,
                                              "yhat_lower": fc.values * 0.9,
                                              "yhat_upper": fc.values * 1.1}),
                "model": "Holt-Winters",
            }
        except Exception as e:
            print(f"HW failed: {e}")

    return None
