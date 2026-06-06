"""Face Registration page - Webcam capture and embedding build."""

import streamlit as st


def render():
    st.markdown("# 📷 Face Registration")
    st.markdown('<div class="section-header">Đăng Ký Khuôn Mặt Nhân Viên</div>', unsafe_allow_html=True)

    col_cam, col_info = st.columns([3, 2])

    with col_cam:
        st.markdown("### 🎥 Webcam Preview")
        camera_input = st.camera_input("Chụp ảnh khuôn mặt", label_visibility="collapsed")

        if camera_input:
            st.image(camera_input, caption="Ảnh đã chụp", use_container_width=True)

    with col_info:
        st.markdown("### 📋 Thông Tin Nhân Viên")
        emp_id = st.text_input("Employee ID *", placeholder="NV001")
        full_name = st.text_input("Họ tên *", placeholder="Nguyễn Văn A")
        department = st.selectbox("Phòng ban", ["IT", "HR", "Finance", "Marketing", "Sales", "Admin"])
        position = st.text_input("Chức vụ", placeholder="Developer")

        st.markdown("---")

        # Quality metrics (simulated for UI)
        st.markdown("### 📊 Chất Lượng Ảnh")
        img_count = st.session_state.get("capture_count", 0)
        st.metric("Ảnh đã thu", f"{img_count} / 100")

        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Blur Score", "—")
        with mc2:
            st.metric("Brightness", "—")

        st.metric("Face Quality", "—")

    st.markdown("---")

    # Action buttons
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        if st.button("▶️ Start Capture", use_container_width=True, type="primary"):
            st.info("💡 Sử dụng terminal: `python -m src.attendance.register_employee --employee_id NV001 --full_name \"Tên\"`")
    with bc2:
        if st.button("⏹️ Stop Capture", use_container_width=True):
            st.session_state["capture_count"] = 0
    with bc3:
        if st.button("🧠 Build Embedding", use_container_width=True):
            st.info("💡 Chạy: `python -m src.attendance.build_embeddings`")

    st.markdown("---")
    st.markdown("""
    > **Hướng dẫn:** 
    > 1. Nhập thông tin nhân viên
    > 2. Bắt đầu capture ảnh qua webcam (quay các góc mặt khác nhau)
    > 3. Sau khi đủ ảnh, nhấn Build Embedding để tạo vector nhận diện
    """)
