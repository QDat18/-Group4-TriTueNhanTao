"""
Face Attendance Management System - Streamlit Frontend
Main entry point with sidebar navigation.
"""

import streamlit as st

st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }

    /* Stat cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
    }
    .stat-card h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .stat-card p {
        font-size: 0.85rem;
        color: #64748b;
        margin: 0;
    }

    .stat-blue h2 { color: #2563eb; }
    .stat-green h2 { color: #10b981; }
    .stat-yellow h2 { color: #d97706; }
    .stat-red h2 { color: #dc2626; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, rgba(37,99,235,0.08), transparent);
        border-left: 4px solid #2563eb;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        color: #0f172a;
    }

    /* Employee card */
    .emp-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .emp-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    }

    /* Camera badge */
    .badge-online {
        display: inline-block;
        background: #10b981;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-offline {
        display: inline-block;
        background: #ef4444;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 0.9rem;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown("## 🧑‍💼 Face Attendance")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "👥 Employees",
            "📷 Face Registration",
            "⏱️ Realtime Attendance",
            "📋 Attendance Logs",
            "🧠 AI System",
            "📈 Reports",
            "🎥 Cameras",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("© 2026 Group 4 - HVNH")

# ── Page Router ──
if page == "📊 Dashboard":
    from pages import dashboard
    dashboard.render()
elif page == "👥 Employees":
    from pages import employees
    employees.render()
elif page == "📷 Face Registration":
    from pages import face_registration
    face_registration.render()
elif page == "⏱️ Realtime Attendance":
    from pages import realtime_attendance
    realtime_attendance.render()
elif page == "📋 Attendance Logs":
    from pages import attendance_logs
    attendance_logs.render()
elif page == "🧠 AI System":
    from pages import ai_system
    ai_system.render()
elif page == "📈 Reports":
    from pages import reports
    reports.render()
elif page == "🎥 Cameras":
    from pages import cameras
    cameras.render()
elif page == "⚙️ Settings":
    from pages import settings
    settings.render()
