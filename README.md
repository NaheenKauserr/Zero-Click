# Zero Click AI — Genesis Training Company

**AI-Powered Company Data Analysis Platform**  
Python & AI Internship · 8th Semester EEE · Karnataka · 11 Interns · 5 Modules

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Run the app
streamlit run app.py
```

---

## Login Credentials

| Username | Password | Role    |
|----------|----------|---------|
| admin    | admin    | Admin   |
| demo     | demo123  | Analyst |

---

## Project Structure

```
├── app.py               ← Main application (entry point)
├── analysis.py          ← Groq AI Advanced Analysis Hub
├── data_ingestion.py    ← File loading (CSV, Excel, JSON)
├── data_cleaning.py     ← Auto-cleaning with report
├── data_analysis.py     ← Column types, stats, correlations
├── kpi_generator.py     ← Smart KPI generation
├── insights.py          ← Groq AI insights + rule-based fallback
├── ml_engine.py         ← K-Means, Regression, Classification, Feature Importance
├── forecasting.py       ← Prophet + Holt-Winters forecasting
├── visualization.py     ← All chart generators
├── chatbot.py           ← Groq-powered floating chatbot
├── report_generator.py  ← PDF report generation (fpdf2)
├── utils.py             ← Shared theme + formatting utilities
├── users.json           ← User credentials store
├── requirements.txt
├── .env                 ← GROQ_API_KEY (never commit)
└── .streamlit/
    └── config.toml      ← Streamlit theme config
```

---

## Modules

| Module | Members | Features |
|--------|---------|----------|
| Module 1 — Data Management | Snehal, Ammar, Nazhat | Universal Upload, Smart Profiling, Data Quality |
| Module 2 — NLP Chatbot | Samruddhi, Dhaval | Natural Language Queries, Context Memory, Suggestions |
| Module 3 — Analytics | Keerti, Vaishnavi | AI Insights, Predictive Engine, Recommendations |
| Module 4 — Dashboard | NK (Lead), Yusuf, Naheen | Glass UI, Visualizations, Real-Time Refresh, PDF Reports |
| Module 5 — Integration | Manu, Anoosha | RBAC, Voice Chatbot, Email Scheduler |

---

## Tech Stack

- **Frontend**: Streamlit, Plotly, Custom CSS (Glassmorphism)
- **AI/LLM**: Groq API (llama-3.1-8b, llama-3.3-70b)
- **ML**: scikit-learn, statsmodels, Prophet
- **PDF**: fpdf2
- **Data**: Pandas, NumPy

---

*Genesis Training Company · Zero Click AI Platform*
