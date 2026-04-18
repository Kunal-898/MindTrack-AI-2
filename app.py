"""
MINDTRACK - AI-Driven Mental Health Sentiment & Trend Analysis
Main Streamlit Application Entry Point
"""

import streamlit as st

st.set_page_config(
    page_title="MindTrack",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject custom CSS matching the HTML UI's glassmorphism style ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* Gradient background */
.stApp {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 50%, #EFF6FF 100%);
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.5);
    border-radius: 24px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #10B981);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; color: white; }

/* Emotion pill badges */
.emotion-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    margin: 2px;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* Sidebar nav link style */
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Session state bootstrap ────────────────────────────────────────────────────
for key, default in {
    "logged_in": False,
    "username": "",
    "page": "login",
    "journal_entries": [],    # list of dicts: {text, emotions, timestamp, wellness}
    "analysis_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Router ─────────────────────────────────────────────────────────────────────
from pages_ui.login_page      import render_login
from pages_ui.dashboard_page  import render_dashboard
from pages_ui.analytics_page  import render_analytics
from pages_ui.settings_page   import render_settings

if not st.session_state.logged_in:
    render_login()
else:
    page = st.session_state.page
    if   page == "dashboard":  render_dashboard()
    elif page == "analytics":  render_analytics()
    elif page == "settings":   render_settings()
    else:
        st.session_state.page = "dashboard"
        render_dashboard()
