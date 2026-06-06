"""Dashboard page - Overview statistics and charts."""

import streamlit as st
import pandas as pd
from api_client import get_dashboard_stats, get_attendance_chart, get_department_ranking, list_devices


def render():
    st.markdown("# 📊 Dashboard")
    st.markdown('<div class="section-header">Tổng Quan Hệ Thống</div>', unsafe_allow_html=True)

    # ── Stats Cards ──
    stats = get_dashboard_stats()
    if "error" in stats:
        st.error(f"⚠️ {stats['error']}")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'''
        <div class="stat-card stat-blue">
            <p>👥 Tổng Nhân Viên</p>
            <h2>{stats["total_employees"]}</h2>
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
        <div class="stat-card stat-green">
            <p>✅ Đi Làm Hôm Nay</p>
            <h2>{stats["present_today"]}</h2>
        </div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
        <div class="stat-card stat-yellow">
            <p>⏰ Đi Muộn</p>
            <h2>{stats["late_today"]}</h2>
        </div>''', unsafe_allow_html=True)
    with c4:
        st.markdown(f'''
        <div class="stat-card stat-red">
            <p>❌ Vắng Mặt</p>
            <h2>{stats["absent_today"]}</h2>
        </div>''', unsafe_allow_html=True)

    st.markdown("")

    # ── Attendance Chart ──
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-header">📈 Biểu Đồ Chấm Công 30 Ngày</div>', unsafe_allow_html=True)
        chart_data = get_attendance_chart(30)
        if chart_data.get("data"):
            df = pd.DataFrame(chart_data["data"])
            df["date"] = pd.to_datetime(df["date"])
            st.area_chart(df.set_index("date")["count"], color="#6366f1", use_container_width=True)
        else:
            st.info("Chưa có dữ liệu chấm công.")

    with col_right:
        st.markdown('<div class="section-header">🏆 Phòng Ban Đúng Giờ</div>', unsafe_allow_html=True)
        ranking = get_department_ranking()
        if ranking.get("data"):
            for dept in ranking["data"][:5]:
                st.markdown(f"""
                **{dept['department']}** — {dept['rate']}%
                """)
                st.progress(dept["rate"] / 100)
        else:
            st.info("Chưa có dữ liệu phòng ban.")

    # ── Camera Status ──
    st.markdown('<div class="section-header">🎥 Camera Đang Hoạt Động</div>', unsafe_allow_html=True)
    devices = list_devices()
    if devices.get("data"):
        cols = st.columns(min(4, len(devices["data"])))
        for i, dev in enumerate(devices["data"]):
            with cols[i % len(cols)]:
                badge = "badge-online" if dev["is_active"] else "badge-offline"
                status = "Online" if dev["is_active"] else "Offline"
                st.markdown(f'''
                <div class="emp-card">
                    <strong>{dev["device_id"]}</strong> - {dev.get("device_name", "")}
                    <br/><small>{dev.get("location", "")}</small>
                    <br/><span class="{badge}">{status}</span>
                </div>''', unsafe_allow_html=True)
    else:
        st.info("Chưa có camera nào được cấu hình.")
