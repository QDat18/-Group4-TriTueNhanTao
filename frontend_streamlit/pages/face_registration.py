"""Face Registration page - Webcam capture and embedding build."""

import streamlit as st
import time
import requests
from api_client import (
    list_employees,
    list_embeddings,
    create_employee,
    start_registration,
    get_registration_progress,
    stop_registration,
    rebuild_embeddings,
    API_BASE
)

BACKEND_URL = API_BASE.replace("/api", "")


def render():
    st.markdown("# 📷 Face Registration")
    st.markdown('<div class="section-header">Đăng Ký Khuôn Mặt Nhân Viên</div>', unsafe_allow_html=True)

    # ── Fetch Existing Data ──
    emps_data = list_employees()
    embs_data = list_embeddings()

    if "error" in emps_data or "error" in embs_data:
        st.error("⚠️ Không thể kết nối tới máy chủ API. Vui lòng chạy backend FastAPI trước.")
        return

    employees = emps_data.get("data", [])
    embeddings = embs_data.get("data", [])
    registered_ids = {emb["employee_id"] for emb in embeddings}

    # ── Step 1: Select or Create Employee ──
    st.markdown("### 👤 Bước 1: Chọn hoặc nhập thông tin Nhân viên")
    
    # Filter employees who don't have face embeddings yet
    unregistered_employees = [e for e in employees if e["employee_id"] not in registered_ids]
    
    options = ["➕ Thêm nhân viên mới thủ công"] + [
        f"{e['employee_id']} - {e['full_name']} ({e.get('department', 'IT')})" for e in unregistered_employees
    ]

    selected_option = st.selectbox("Chọn nhân viên đăng ký mặt", options)

    is_manual = selected_option == "➕ Thêm nhân viên mới thủ công"
    
    if is_manual:
        col1, col2 = st.columns(2)
        with col1:
            emp_id = st.text_input("Mã Nhân Viên *", placeholder="NV001").strip()
            dept = st.selectbox("Phòng ban", ["IT", "HR", "Finance", "Marketing", "Sales", "Admin"])
        with col2:
            name = st.text_input("Họ tên nhân viên *", placeholder="Nguyễn Văn A").strip()
            position = st.text_input("Chức vụ", placeholder="Developer").strip()
        email = st.text_input("Email", placeholder="email@company.com").strip()
        phone = st.text_input("SĐT", placeholder="0901234567").strip()
    else:
        # Extract ID from choice
        emp_id = selected_option.split(" - ")[0]
        emp = next((e for e in employees if e["employee_id"] == emp_id), None)
        name = emp["full_name"] if emp else ""
        dept = emp["department"] if emp else "IT"
        position = emp["position"] if emp else ""
        email = emp.get("email", "")
        phone = emp.get("phone", "")
        
        st.info(f"Đang chọn: **{emp_id}** - Họ tên: **{name}** (Phòng ban: {dept})")

    st.markdown("---")

    # ── Step 2: Camera Capture Flow ──
    st.markdown("### 🎥 Bước 2: Chụp 50 ảnh mẫu khuôn mặt")
    
    # Session state for active capture session
    is_capturing = st.session_state.get("reg_capturing", False)
    active_emp_id = st.session_state.get("reg_emp_id", "")

    # Progress placeholder
    progress_placeholder = st.empty()
    gallery_placeholder = st.empty()

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        start_btn = st.button("▶️ Bắt đầu chụp ảnh mẫu", use_container_width=True, type="primary", disabled=is_capturing)
    with col_btn2:
        stop_btn = st.button("⏹️ Dừng chụp", use_container_width=True, disabled=not is_capturing)
    with col_btn3:
        build_btn = st.button("🧠 Tạo Vector Nhận Diện", use_container_width=True, disabled=is_capturing)

    # Handle Actions
    if start_btn:
        if not emp_id or not name:
            st.warning("⚠️ Vui lòng nhập Mã NV và Họ tên trước khi chụp.")
        else:
            # Create employee record if manual input
            if is_manual:
                exists = any(e["employee_id"] == emp_id for e in employees)
                if not exists:
                    res = create_employee({
                        "employee_id": emp_id,
                        "full_name": name,
                        "department": dept,
                        "position": position or "Employee",
                        "email": email or None,
                        "phone": phone or None
                    })
                    if "error" in res:
                        st.error(f"Lỗi tạo nhân viên: {res['error']}")
                        st.stop()
            
            # Start background thread webcam capture
            payload = {
                "employee_id": emp_id,
                "full_name": name,
                "max_images": 50
            }
            res = start_registration(payload)
            if "error" in res:
                st.error(f"❌ Không thể bắt đầu chụp: {res['error']}")
            else:
                st.session_state["reg_capturing"] = True
                st.session_state["reg_emp_id"] = emp_id
                st.success(f"🎥 Đang khởi động camera máy chủ. Vui lòng nhìn vào camera và quay nhẹ đầu.")
                time.sleep(1.0)
                st.rerun()

    if stop_btn:
        stop_registration()
        st.session_state["reg_capturing"] = False
        st.warning("⏹️ Đã gửi lệnh dừng chụp ảnh mẫu.")
        st.rerun()

    if build_btn:
        with st.spinner("Đang tính toán các vector đặc trưng khuôn mặt và cập nhật CSDL Supabase..."):
            res = rebuild_embeddings()
            if "error" in res:
                st.error(f"❌ Lỗi tạo vector: {res['error']}")
            else:
                st.success("🎉 Tạo vector nhận diện thành công! Nhân viên đã có thể điểm danh.")
                st.balloons()
                time.sleep(2)
                st.rerun()

    # Polling capture progress
    if is_capturing and active_emp_id:
        # Fetch progress
        prog = get_registration_progress(active_emp_id)
        if "error" in prog:
            st.error(f"Lỗi tiến trình: {prog['error']}")
            st.session_state["reg_capturing"] = False
            st.rerun()
        else:
            count = min(prog.get("count", 0), 50)
            is_running = prog.get("is_running", False)

            with progress_placeholder.container():
                st.progress(count / 50.0)
                st.metric("Số lượng ảnh mẫu đã chụp", f"{count} / 50")
                if is_running:
                    st.info("ℹ️ Nhìn thẳng vào webcam và xoay nhẹ đầu (lên, xuống, trái, phải). Cửa sổ OpenCV đang mở.")
                else:
                    st.success("✅ Chụp hoàn tất! Hãy nhấn nút 'Tạo Vector Nhận Diện' phía trên.")
                    st.session_state["reg_capturing"] = False

            # Display a small gallery of captured images
            with gallery_placeholder.container():
                if count > 0:
                    st.markdown("#### Gallery ảnh mẫu vừa chụp")
                    cols = st.columns(5)
                    for idx in range(min(5, count)):
                        img_url = f"{BACKEND_URL}/api/portraits/{active_emp_id}/{active_emp_id}_{idx:03d}.jpg"
                        with cols[idx % 5]:
                            st.image(img_url, use_container_width=True)

            if is_running and count < 50:
                time.sleep(1.2)
                st.rerun()
