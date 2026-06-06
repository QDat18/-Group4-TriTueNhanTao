"""Attendance Logs page - Filterable attendance history."""

import streamlit as st
import pandas as pd
from datetime import date
from api_client import get_attendance_logs


def render():
    st.markdown("# 📋 Attendance Logs")
    st.markdown('<div class="section-header">Nhật Ký Chấm Công</div>', unsafe_allow_html=True)

    # ── Filters ──
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_date = st.date_input("📅 Ngày", value=date.today())
    with c2:
        dept_filter = st.selectbox("🏢 Phòng ban", ["Tất cả", "IT", "HR", "Finance", "Marketing", "Sales"])
    with c3:
        emp_filter = st.text_input("👤 Mã NV", placeholder="NV001")

    # ── Fetch Data ──
    params = {"limit": 200}
    if selected_date:
        params["date"] = selected_date.isoformat()
    if dept_filter != "Tất cả":
        params["department"] = dept_filter
    if emp_filter:
        params["employee_id"] = emp_filter

    logs = get_attendance_logs(**params)

    if "error" in logs:
        st.error(f"⚠️ {logs['error']}")
        return

    data = logs.get("data", [])

    if not data:
        st.info("Không có bản ghi chấm công cho bộ lọc đã chọn.")
        return

    # ── Display Table ──
    df = pd.DataFrame(data)

    # Format columns
    display_cols = ["check_time", "employee_id", "full_name", "department", "similarity", "status", "camera_id"]
    available_cols = [c for c in display_cols if c in df.columns]
    df_display = df[available_cols].copy()

    if "check_time" in df_display.columns:
        df_display["check_time"] = df_display["check_time"].str[:19].str.replace("T", " ")

    if "similarity" in df_display.columns:
        df_display["similarity"] = df_display["similarity"].apply(lambda x: f"{x:.2f}" if x else "—")

    # Rename columns for display
    col_map = {
        "check_time": "⏰ Thời gian",
        "employee_id": "🆔 Mã NV",
        "full_name": "👤 Họ tên",
        "department": "🏢 Phòng ban",
        "similarity": "📊 Similarity",
        "status": "✅ Trạng thái",
        "camera_id": "🎥 Camera",
    }
    df_display.rename(columns=col_map, inplace=True)

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.caption(f"Tổng: {len(data)} bản ghi")

    # ── Export ──
    if st.button("📥 Xuất CSV"):
        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Tải xuống", csv, f"attendance_{selected_date}.csv", "text/csv")
