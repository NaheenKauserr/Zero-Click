"""chatbot.py — Groq-powered floating chatbot with chart generation."""
import os
import re
import streamlit as st
import pandas as pd

def chat_response(query: str, df: pd.DataFrame, column_types: dict) -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "")

    if df is None or df.empty:
        return {"text": "No dataset is loaded. Please upload a file first.", "code": None}

    if not api_key or api_key == "your_key_here":
        return _fallback_response(query, df, column_types)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        system = (
            "You are a Data Analyst AI for a glassmorphism dashboard. "
            "Answer questions about the dataset. "
            "If the user asks for a chart, write Python code using 'import plotly.express as px' "
            "creating a figure named `fig`. Wrap code in ```python blocks. "
            "Apply this theme to all figs: "
            "fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', "
            "font=dict(color='rgba(255,255,255,0.75)')); "
            "fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')"
        )
        context = (
            f"Dataset: {len(df)} rows, columns={list(df.columns)}\n"
            f"Sample:\n{df.head(5).to_csv(index=False)}\n"
            f"User: {query}"
        )
        resp = client.chat.completions.create(
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": context}],
            model="llama-3.1-8b-instant",
        )
        text = resp.choices[0].message.content
        code_match = re.search(r"```python(.*?)```", text, re.DOTALL)
        code = None
        if code_match:
            code = code_match.group(1).strip()
            text = text.replace(code_match.group(0), "").strip() or "Here's the visualization:"
        return {"text": text, "code": code}

    except Exception as e:
        return _fallback_response(query, df, column_types)

def _fallback_response(query: str, df: pd.DataFrame, column_types: dict) -> dict:
    q = query.lower()
    num_cols = column_types.get("numeric", [])
    cat_cols = column_types.get("categorical", [])

    if any(w in q for w in ["row", "size", "shape", "how many"]):
        return {"text": f"The dataset has **{len(df):,} rows** and **{len(df.columns)} columns**.", "code": None}
    if any(w in q for w in ["column", "feature", "field"]):
        return {"text": f"Columns: {', '.join(f'`{c}`' for c in df.columns)}", "code": None}
    if any(w in q for w in ["missing", "null", "nan"]):
        m = int(df.isnull().sum().sum())
        return {"text": f"There are **{m:,}** missing values across the dataset.", "code": None}
    if num_cols and any(w in q for w in ["total", "sum", "revenue", "sales"]):
        col = num_cols[0]
        for c in num_cols:
            if any(w in c.lower() for w in ["sales", "revenue", "amount"]):
                col = c; break
        total = df[col].sum()
        return {"text": f"Total **{col}**: **{total:,.2f}**", "code": None}
    if num_cols and any(w in q for w in ["average", "mean", "avg"]):
        col = num_cols[0]
        return {"text": f"Average **{col}**: **{df[col].mean():,.2f}**", "code": None}
    if cat_cols and any(w in q for w in ["categor", "segment", "region", "top"]):
        col = cat_cols[0]
        top = df[col].value_counts().head(3)
        resp = f"Top values in **{col}**:\n\n" + "\n".join(f"- {v}: {c}" for v, c in top.items())
        return {"text": resp, "code": None}

    return {
        "text": (
            f"Your dataset has {len(df):,} rows. "
            f"I can help you analyze **{', '.join(num_cols[:3])}** and other columns. "
            "Try asking about totals, averages, trends, or request a specific chart!"
        ),
        "code": None,
    }
