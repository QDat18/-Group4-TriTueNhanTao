"""Realtime Attendance page."""

import streamlit as st
import time
import requests
from api_client import API_BASE, get_attendance_logs

BACKEND_URL = API_BASE.replace("/api", "")


def get_current_face():
    try:
        resp = requests.get(f"{BACKEND_URL}/api/attendance/current-face", timeout=2)
        if resp.status_code == 200:
            return resp.json().get("data")
    except Exception:
        pass
    return None


def render():
    st.markdown("# ⏱️ Realtime Attendance")
    st.markdown('<div class="section-header">Chấm Công Thời Gian Thực</div>', unsafe_allow_html=True)

    col_cam, col_result = st.columns([3, 2])

    with col_cam:
        st.markdown("### 🎥 Live Camera Stream")
        
        # Display the live MJPEG stream from the FastAPI backend
        stream_url = f"{BACKEND_URL}/api/attendance/stream"
        
        st.markdown(f"""
        <div style="background: #111; border: 3px solid #6366f1; border-radius: 14px; overflow: hidden; padding: 2px;">
            <img src="{stream_url}" style="width: 100%; height: auto; display: block; border-radius: 12px;" 
                 onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800';" />
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        st.markdown("### 🎯 Kết Quả Nhận Diện")
        
        # We use st.empty to show the live detection results
        result_placeholder = st.empty()
        
        # Auto-refresh check
        auto_refresh = st.checkbox("🔄 Tự động cập nhật kết quả", value=True)

    # ── Recent Logs ──
    st.markdown('<div class="section-header">📋 Nhật Ký Điểm Danh Gần Đây</div>', unsafe_allow_html=True)
    logs_placeholder = st.empty()

    def update_logs_display():
        logs = get_attendance_logs(limit=10)
        with logs_placeholder.container():
            if logs.get("data"):
                for log in logs["data"][:8]:
                    time_str = log.get("check_time", "")[:19].replace("T", " ")
                    sim = log.get("similarity", 0)
                    status = log.get("status", "")
                    emp_id = log.get("employee_id", "")
                    name = log.get("full_name", "Unknown")

                    status_icon = "✅" if status in ("SUCCESS", "LATE") else "⏱️" if status == "COOLDOWN" else "❌"
                    status_lbl = "Vào đúng giờ" if status == "SUCCESS" else "Vào muộn" if status == "LATE" else "Đã điểm danh" if status == "COOLDOWN" else status
                    
                    st.markdown(
                        f"`{time_str}` | **{emp_id}** — {name} | Độ tương đồng: `{sim:.2f}` | {status_icon} **{status_lbl}**"
                    )
            else:
                st.info("Chưa có log chấm công hôm nay.")

    # Run polling loop
    if auto_refresh:
        # Run for 20 seconds, polling every 1 second, then reload the page
        for i in range(20):
            face = get_current_face()
            with result_placeholder.container():
                if face:
                    status = face.get("status", "UNKNOWN")
                    name = face.get("full_name", "Unknown")
                    liveness = face.get("liveness_score", 0.0)
                    label = face.get("label", "")
                    
                    # Status styling
                    bg_color = "#e6f4ea" if status == "SUCCESS" else "#fef7e0" if status == "LATE" else "#eef2ff" if status == "COOLDOWN" else "#fce8e6"
                    text_color = "#137333" if status == "SUCCESS" else "#b06000" if status == "LATE" else "#3949ab" if status == "COOLDOWN" else "#c5221f"
                    status_desc = "VÀO ĐÚNG GIỜ" if status == "SUCCESS" else "VÀO MUỘN" if status == "LATE" else "ĐÃ ĐIỂM DANH" if status == "COOLDOWN" else "THẤT BẠI"

                    st.markdown(f"""
                    <div style="background: {bg_color}; border: 2px solid {text_color}44; border-radius: 14px; padding: 1.5rem; text-align: center; color: {text_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                        <p style="font-size: 0.85rem; font-weight: 700; margin: 0 0 8px 0; opacity: 0.85;">NHÂN VIÊN ĐÃ PHÁT HIỆN</p>
                        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: {text_color};">{name}</h2>
                        <div style="display: inline-block; margin-top: 10px; padding: 4px 14px; background: {text_color}18; border-radius: 20px; font-size: 0.75rem; font-weight: 800;">
                            {status_desc}
                        </div>
                        <div style="margin-top: 15px; text-align: left; font-size: 0.82rem; border-top: 1px dashed {text_color}33; padding-top: 10px;">
                            <p style="margin: 4px 0; color: #444;"><strong>Anti-Spoofing:</strong> {liveness * 100:.1f}% (Thật)</p>
                            <p style="margin: 4px 0; color: #444;"><strong>Chi tiết:</strong> {label}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="emp-card" style="text-align: center; padding: 2.2rem 1rem;">
                        <p style="font-size: 2.5rem; margin: 0 0 10px 0;">👤</p>
                        <h4 style="color: #64748b; margin: 0;">Đang chờ người đứng trước camera...</h4>
                        <p style="font-size: 0.78rem; color: #94a3b8; margin: 6px 0 0 0;">Nhìn thẳng vào thiết bị để điểm danh tự động</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            update_logs_display()
            time.sleep(1.0)
            
        # Trigger page rerun to continue polling
        st.rerun()
    else:
        # Non-polling state
        face = get_current_face()
        with result_placeholder.container():
            if face:
                st.write(face)
            else:
                st.info("Nhấp vào 'Refresh' để tải kết quả nhận diện mới nhất.")
        update_logs_display()
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
