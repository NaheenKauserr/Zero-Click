"""analysis.py — Advanced Analysis Hub with Groq AI (from teammate, integrated)."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import utils

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def get_column_types(df):
    num  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat  = df.select_dtypes(include=["object","category"]).columns.tolist()
    dt   = df.select_dtypes(include=["datetime64","datetimetz"]).columns.tolist()
    if not dt:
        for col in list(cat):
            if any(k in col.lower() for k in ["date","time","year","month"]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    if df[col].notna().sum() > 0.5 * len(df):
                        dt.append(col); cat.remove(col)
                except Exception: pass
    return {"numeric": num, "categorical": cat, "datetime": dt}

def compute_advanced_stats(df, types):
    numeric = types["numeric"]
    categorical = types["categorical"]
    kpis = {col: {"avg": df[col].mean(), "sum": df[col].sum(),
                  "min": df[col].min(), "max": df[col].max(),
                  "count": len(df[col].dropna())}
            for col in numeric}
    corr = df[numeric].corr().round(3).to_dict() if len(numeric) >= 2 else {}
    group_impacts = []
    for cat in categorical[:5]:
        for num_col in numeric[:5]:
            try:
                if any(k in cat.lower() for k in ["id","code"]): continue
                groups = df.groupby(cat)[num_col].mean().sort_values(ascending=False)
                if len(groups) < 2: continue
                top_c, top_v   = groups.index[0], float(groups.iloc[0])
                bot_c, bot_v   = groups.index[-1], float(groups.iloc[-1])
                denom = abs(bot_v) if bot_v != 0 else 1
                diff  = abs((top_v - bot_v) / denom) * 100
                group_impacts.append({"cat": cat, "num": num_col,
                                       "top": (top_c, top_v), "bottom": (bot_c, bot_v),
                                       "diff": diff, "means": groups.reset_index().values.tolist()})
            except Exception: continue
    group_impacts.sort(key=lambda x: x["diff"], reverse=True)
    return {"kpis": kpis, "corr": corr, "group_impacts": group_impacts,
            "numeric": numeric, "categorical": categorical, "datetime": types["datetime"]}

def generate_local_insights(stats):
    lines = []
    numeric_cols = stats["numeric"]
    strong = [(a, b, stats["corr"].get(a, {}).get(b, 0))
              for i, a in enumerate(numeric_cols) for b in numeric_cols[i+1:]
              if abs(stats["corr"].get(a, {}).get(b, 0)) > 0.5]
    if strong:
        for a, b, r in strong[:3]:
            lines.append(f"**{a}** and **{b}** show a {'positive' if r>0 else 'negative'} correlation (r={r:.2f})")
    else:
        lines.append("No strong numeric correlations detected.")
    if stats["group_impacts"]:
        for g in stats["group_impacts"][:2]:
            lines.append(f"**{g['cat']}** influences **{g['num']}**: '{g['top'][0]}' leads by {g['diff']:.1f}%")
    lines.append(f"Dataset: {len(stats['numeric'])} metrics · {len(stats['categorical'])} categories")
    return "\n\n".join(f"- {l}" for l in lines)

def get_ai_insights(df, stats):
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY","")
    if not api_key:
        return generate_local_insights(stats)
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        numeric_cols = stats["numeric"]
        num_summary = "\n".join(f"- {h}: avg={v['avg']:.2f}, min={v['min']:.2f}, max={v['max']:.2f}"
                                for h, v in list(stats["kpis"].items())[:6])
        corr_lines = [f"{a} ↔ {b}: {stats['corr'].get(a,{}).get(b,0):.2f}"
                      for i, a in enumerate(numeric_cols)
                      for b in numeric_cols[i+1:]
                      if abs(stats["corr"].get(a,{}).get(b,0)) > 0.4]
        grp_lines = [f"{g['cat']}→{g['num']}: '{g['top'][0]}' vs '{g['bottom'][0]}' diff={g['diff']:.0f}%"
                     for g in stats["group_impacts"][:4]]
        prompt = (f"Senior analyst. Dataset {len(df)}×{len(df.columns)}.\n"
                  f"Numeric stats:\n{num_summary}\n"
                  f"Correlations: {', '.join(corr_lines) or 'None'}\n"
                  f"Drivers: {'; '.join(grp_lines) or 'None'}\n"
                  "Provide: 1) Key Relationships (2-3 bullets) "
                  "2) Important Features (2 bullets) "
                  "3) Patterns (2 bullets) "
                  "4) Summary (2 sentences). Mention column names. Markdown.")
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}], max_tokens=800)
        return resp.choices[0].message.content
    except Exception:
        return generate_local_insights(stats)

def render_advanced_analysis(df):
    if df is None:
        st.info("Upload a dataset to begin analysis.")
        return

    with st.spinner("Deep analysis in progress…"):
        types = get_column_types(df)
        stats = compute_advanced_stats(df, types)

    tabs = st.tabs(["🏠 Overview","🤖 AI Insights","🔗 Correlations",
                    "📈 Scatter","📊 Category Impact","🥧 Distributions"])

    with tabs[0]:
        cols = st.columns(4)
        for i, col_name in enumerate(stats["numeric"][:4]):
            with cols[i % 4]:
                k = stats["kpis"][col_name]
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.7);padding:12px;border-radius:12px;
                            border:1px solid rgba(99,102,241,0.2);text-align:center">
                  <div style="font-size:10px;color:#94a3b8;text-transform:uppercase">{col_name}</div>
                  <div style="font-size:1.4rem;font-weight:700;color:#6366f1;margin:4px 0">
                    {utils.format_number(k['avg'])}</div>
                  <div style="font-size:9px;color:#64748b">Avg · {k['count']} samples</div>
                </div>""", unsafe_allow_html=True)
        if stats["group_impacts"]:
            st.markdown("##### Top Performance Drivers")
            for g in stats["group_impacts"][:3]:
                st.markdown(f"**{g['cat']}** → {g['num']}: "
                            f"_{g['top'][0]}_ leads by **{g['diff']:.1f}%**")

    with tabs[1]:
        if "ai_cache" not in st.session_state or st.session_state.get("_last_df_id") != id(df):
            with st.spinner("Consulting Groq LLM…"):
                st.session_state.ai_cache = get_ai_insights(df, stats)
                st.session_state["_last_df_id"] = id(df)
        st.markdown(st.session_state.ai_cache)

    with tabs[2]:
        if len(stats["numeric"]) >= 2:
            corr_df = df[stats["numeric"]].corr()
            fig = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, title="Correlation Heatmap")
            utils.apply_genesis_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        numeric_cols = stats["numeric"]
        strong = [(a, b, stats["corr"].get(a,{}).get(b,0))
                  for i, a in enumerate(numeric_cols)
                  for b in numeric_cols[i+1:]
                  if abs(stats["corr"].get(a,{}).get(b,0)) > 0.4]
        if strong:
            cols_w = st.columns(2)
            for i, (a, b, r) in enumerate(strong[:4]):
                with cols_w[i % 2]:
                    try:
                        fig = px.scatter(df, x=a, y=b, trendline="ols",
                                         title=f"{a} vs {b} (r={r:.2f})",
                                         color_discrete_sequence=["#6366f1"])
                        utils.apply_genesis_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception: pass

    with tabs[4]:
        if stats["group_impacts"]:
            cols_w = st.columns(2)
            for i, g in enumerate(stats["group_impacts"][:4]):
                with cols_w[i % 2]:
                    bar_df = pd.DataFrame(g["means"], columns=[g["cat"], "Mean Value"])
                    fig = px.bar(bar_df.head(10), x=g["cat"], y="Mean Value",
                                 title=f"Avg {g['num']} by {g['cat']}",
                                 color="Mean Value", color_continuous_scale="Blues")
                    utils.apply_genesis_theme(fig)
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        if stats["categorical"]:
            cols_w = st.columns(2)
            for i, col_name in enumerate(stats["categorical"][:4]):
                with cols_w[i % 2]:
                    counts = df[col_name].value_counts().reset_index()
                    counts.columns = [col_name, "Count"]
                    fig = px.pie(counts.head(8), values="Count", names=col_name,
                                 title=f"Segments: {col_name}", hole=0.4,
                                 color_discrete_sequence=px.colors.sequential.Purples)
                    utils.apply_genesis_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)
