"""Settings page - System configuration."""

import streamlit as st
import datetime
from api_client import get_settings, update_settings, get_model_info


def render():
    st.markdown("# ⚙️ Settings")
    st.markdown('<div class="section-header">Cấu Hình Hệ Thống</div>', unsafe_allow_html=True)

    # ── Fetch Data ──
    cfg = get_settings()
    if "error" in cfg:
        st.error(f"⚠️ Không tải được cấu hình hệ thống: {cfg['error']}")
        return

    # Fetch Model Info
    model_info = get_model_info()

    # ── Build Form ──
    with st.form("settings_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 🎯 Nhận diện khuôn mặt")
            threshold = st.slider(
                "Ngưỡng Similarity nhận diện",
                min_value=0.10,
                max_value=0.99,
                value=float(cfg.get("recognition_threshold", 0.45)),
                step=0.01,
                help="Độ tương đồng cosine tối thiểu để xác thực danh tính (khuyến nghị: 0.40 - 0.50)"
            )
            cooldown = st.number_input(
                "Thời gian chờ Cooldown (giây)",
                min_value=10,
                max_value=86400,
                value=int(cfg.get("cooldown_seconds", 43200)),
                help="Giãn cách tối thiểu giữa 2 lần nhận diện của cùng một người để tránh trùng lặp log"
            )

            st.markdown("---")

            st.markdown("### 🕐 Giờ làm việc & Đi muộn")
            
            # Parse time string safely
            time_str = cfg.get("work_start_time", "08:00")
            try:
                time_parts = list(map(int, time_str.split(":")))
                time_val = datetime.time(time_parts[0], time_parts[1])
            except Exception:
                time_val = datetime.time(8, 0)

            work_start = st.time_input(
                "Giờ bắt đầu làm việc",
                value=time_val
            )
            allow_late = st.number_input(
                "Phút ân hạn đi trễ",
                min_value=0,
                max_value=180,
                value=int(cfg.get("allow_late_minutes", 30))
            )
            st.caption("Ví dụ: Giờ vào là 08:00 và ân hạn 30 phút, nhân viên điểm danh sau 08:30 sẽ bị tính là Đi muộn (LATE).")

        with col_right:
            st.markdown("### 📹 Cấu hình Camera điểm danh")
            source_type = st.selectbox(
                "Nguồn camera",
                ["webcam", "ip_camera"],
                index=0 if cfg.get("camera_source_type") == "webcam" else 1,
                format_func=lambda x: "Webcam máy tính / Thiết bị gắn ngoài" if x == "webcam" else "Điện thoại / IP Camera (Dùng địa chỉ URL)"
            )

            webcam_index = st.selectbox(
                "Số hiệu cổng Webcam (Index)",
                [0, 1, 2, 3],
                index=int(cfg.get("camera_webcam_index", 0)),
                help="Chỉ số USB webcam kết nối với máy tính"
            )

            ip_url = st.text_input(
                "Địa chỉ IP Stream (URL)",
                value=cfg.get("camera_ip_url", ""),
                placeholder="ví dụ: http://192.168.1.5:8080/video",
                help="Cổng phát video của app camera điện thoại qua Wi-Fi"
            )

            st.markdown("---")

            # ── Database Info ──
            st.markdown("### 🗄️ Cơ sở dữ liệu")
            # Parse host name from URL if possible
            db_host = "Supabase Cloud"
            st.markdown(f"""
            <div class="emp-card" style="line-height:1.8; font-size:0.88rem;">
                <p><strong>Nhà cung cấp:</strong> Supabase Cloud</p>
                <p><strong>Khu vực:</strong> Singapore (ap-southeast-1)</p>
                <p><strong>Các bảng:</strong> employees, face_embeddings, attendance_logs, devices</p>
            </div>
            """, unsafe_allow_html=True)

        # Form submission
        submitted = st.form_submit_button("💾 Lưu Cấu Hình Hệ Thống", use_container_width=True, type="primary")
        if submitted:
            # Format time to HH:MM
            formatted_time = work_start.strftime("%H:%M")
            payload = {
                "work_start_time": formatted_time,
                "allow_late_minutes": int(allow_late),
                "cooldown_seconds": int(cooldown),
                "recognition_threshold": float(threshold),
                "camera_source_type": source_type,
                "camera_webcam_index": int(webcam_index),
                "camera_ip_url": ip_url
            }

            res = update_settings(payload)
            if "error" in res:
                st.error(f"❌ Không thể cập nhật cấu hình: {res['error']}")
            else:
                st.success("✅ Cấu hình hệ thống đã được cập nhật thành công!")
                st.rerun()

    # ── Model Info (outside form) ──
    st.markdown("---")
    st.markdown("### 🤖 Thông tin Mô hình Trí tuệ Nhân tạo")
    if "error" not in model_info:
        model_name = model_info.get("model_name", "Unknown")
        metrics = model_info.get("metrics", {})
        
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric("Model Path", model_name)
        with c_m2:
            st.metric("Rank-1 Accuracy", metrics.get("rank1", "—"))
        with c_m3:
            st.metric("Rank-5 Accuracy", metrics.get("rank5", "—"))
        with c_m4:
            st.metric("Equal Error Rate (EER)", metrics.get("eer", "—"))
            
        st.markdown(f"Ngưỡng tối ưu: `{metrics.get('threshold', '—')}` | Trình dò khuôn mặt: `InsightFace (Buffalo_L)`")
        
        # Display Markdown Report if loaded
        report_content = model_info.get("report", "")
        if report_content:
            st.markdown("---")
            st.markdown("#### 📋 Báo cáo đánh giá chi tiết (AFDB Masked Face)")
            with st.expander("Xem báo cáo chi tiết", expanded=False):
                st.markdown(report_content)
    else:
        st.info("Không lấy được thông tin mô hình từ API.")
