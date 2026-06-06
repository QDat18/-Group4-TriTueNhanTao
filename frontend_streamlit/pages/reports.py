"""Reports page - Attendance analytics and charts."""

import streamlit as st
import pandas as pd
from api_client import get_report_summary, get_report_by_department


def render():
    st.markdown("# 📈 Reports")
    st.markdown('<div class="section-header">Báo Cáo Chấm Công</div>', unsafe_allow_html=True)

    # ── Period Selector ──
    period = st.radio("Khoảng thời gian", ["day", "week", "month"], horizontal=True,
                      format_func=lambda x: {"day": "📅 Ngày", "week": "📆 Tuần", "month": "🗓️ Tháng"}[x])

    # ── Summary ──
    summary = get_report_summary(period)
    if "error" in summary:
        st.error(f"⚠️ {summary['error']}")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'''
        <div class="stat-card stat-green">
            <p>Tỷ Lệ Đi Làm</p>
            <h2>{summary.get("attendance_rate", 0)}%</h2>
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
        <div class="stat-card stat-yellow">
            <p>Tỷ Lệ Đi Muộn</p>
            <h2>{summary.get("late_rate", 0)}%</h2>
        </div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
        <div class="stat-card stat-red">
            <p>Tỷ Lệ Vắng Mặt</p>
            <h2>{summary.get("absent_rate", 0)}%</h2>
        </div>''', unsafe_allow_html=True)

    st.markdown("")

    # ── Daily Chart ──
    st.markdown('<div class="section-header">📊 Biểu Đồ Theo Ngày</div>', unsafe_allow_html=True)
    daily = summary.get("daily_data", [])
    if daily:
        df = pd.DataFrame(daily)
        df["date"] = pd.to_datetime(df["date"])
        st.bar_chart(df.set_index("date")["present"], color="#6366f1", use_container_width=True)
    else:
        st.info("Chưa có dữ liệu.")

    # ── Department Chart ──
    st.markdown('<div class="section-header">🏢 Biểu Đồ Theo Phòng Ban</div>', unsafe_allow_html=True)
    dept_data = get_report_by_department()
    if dept_data.get("data"):
        df_dept = pd.DataFrame(dept_data["data"])
        if not df_dept.empty:
            st.bar_chart(df_dept.set_index("department")[["present", "absent"]], use_container_width=True)
    else:
        st.info("Chưa có dữ liệu phòng ban.")
