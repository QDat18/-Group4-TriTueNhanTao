"""Realtime Attendance page."""

import streamlit as st
from api_client import get_attendance_logs


def render():
    st.markdown("# ⏱️ Realtime Attendance")
    st.markdown('<div class="section-header">Chấm Công Thời Gian Thực</div>', unsafe_allow_html=True)

    col_cam, col_result = st.columns([3, 2])

    with col_cam:
        st.markdown("### 🎥 Camera Feed")
        st.markdown("""
        <div style="background: #1a1a2e; border: 2px dashed #6366f1; border-radius: 12px; 
                    padding: 4rem 2rem; text-align: center; color: #888;">
            <p style="font-size: 3rem;">📹</p>
            <p>Camera Preview</p>
            <p style="font-size: 0.8rem;">Chạy: <code>python -m src.attendance.realtime_recognition</code></p>
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        st.markdown("### 🎯 Kết Quả Nhận Diện")

        st.markdown("""
        <div class="emp-card" style="text-align: center;">
            <p style="font-size: 1.5rem; font-weight: 700; color: #60a5fa;">—</p>
            <p style="opacity: 0.7;">Đang chờ nhận diện...</p>
            <br/>
            <p>Similarity: <strong>—</strong></p>
            <p>Status: <strong>—</strong></p>
            <p>Time: <strong>—</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # ── Recent Logs ──
    st.markdown('<div class="section-header">📋 Log Thời Gian Thực</div>', unsafe_allow_html=True)

    logs = get_attendance_logs(limit=20)
    if logs.get("data"):
        for log in logs["data"][:10]:
            time_str = log.get("check_time", "")[:19].replace("T", " ")
            sim = log.get("similarity", 0)
            status = log.get("status", "")
            emp_id = log.get("employee_id", "")
            name = log.get("full_name", "Unknown")

            status_icon = "✅" if status == "SUCCESS" else "❌"
            st.markdown(
                f"`{time_str}` | **{emp_id}** {name} | Sim: `{sim:.2f}` | {status_icon} {status}"
            )
    else:
        st.info("Chưa có log chấm công.")

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
