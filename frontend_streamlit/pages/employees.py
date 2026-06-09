"""Employees page - CRUD operations for employee management."""

import streamlit as st
from api_client import list_employees, create_employee, update_employee, delete_employee, get_employee, delete_embedding, delete_portraits


def render():
    st.markdown("# 👥 Employee Management")

    # ── Toolbar ──
    col_search, col_dept, col_add = st.columns([3, 2, 1])
    with col_search:
        search = st.text_input("🔍 Tìm kiếm", placeholder="Mã NV hoặc tên...")
    with col_dept:
        department = st.selectbox("Phòng ban", ["Tất cả", "IT", "HR", "Finance", "Marketing", "Sales", "Admin"])
    with col_add:
        st.markdown("<br/>", unsafe_allow_html=True)
        add_btn = st.button("➕ Thêm NV", use_container_width=True)

    # ── Add Employee Modal ──
    if add_btn:
        st.session_state["show_add_form"] = True

    if st.session_state.get("show_add_form"):
        st.markdown('<div class="section-header">Thêm Nhân Viên Mới</div>', unsafe_allow_html=True)
        with st.form("add_employee_form"):
            c1, c2 = st.columns(2)
            with c1:
                emp_id = st.text_input("Mã NV *", placeholder="NV001")
                dept = st.selectbox("Phòng ban", ["IT", "HR", "Finance", "Marketing", "Sales", "Admin"])
            with c2:
                name = st.text_input("Họ tên *", placeholder="Nguyễn Văn A")
                position = st.text_input("Chức vụ", placeholder="Developer")

            email = st.text_input("Email", placeholder="email@company.com")
            phone = st.text_input("SĐT", placeholder="0901234567")

            submitted = st.form_submit_button("💾 Lưu", use_container_width=True)
            if submitted:
                if emp_id and name:
                    result = create_employee({
                        "employee_id": emp_id,
                        "full_name": name,
                        "department": dept,
                        "position": position or "Employee",
                        "email": email or None,
                        "phone": phone or None,
                    })
                    if "error" in result:
                        st.error(f"Lỗi: {result['error']}")
                    else:
                        st.success(f"✅ Đã tạo nhân viên {emp_id}")
                        st.session_state["show_add_form"] = False
                        st.rerun()
                else:
                    st.warning("Vui lòng nhập Mã NV và Họ tên")

    # ── Employee List ──
    st.markdown('<div class="section-header">Danh Sách Nhân Viên</div>', unsafe_allow_html=True)

    dept_filter = None if department == "Tất cả" else department
    data = list_employees(search=search or None, department=dept_filter)

    if "error" in data:
        st.error(f"⚠️ {data['error']}")
        return

    employees = data.get("data", [])

    if not employees:
        st.info("Không tìm thấy nhân viên nào.")
        return

    for emp in employees:
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([1.5, 2, 1.5, 1.5, 2])
            with c1:
                st.markdown(f"**{emp['employee_id']}**")
            with c2:
                st.markdown(emp.get("full_name", "—"))
            with c3:
                st.markdown(f"🏢 {emp.get('department', '—')}")
            with c4:
                st.markdown(f"💼 {emp.get('position', '—')}")
            with c5:
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button("📝", key=f"edit_{emp['employee_id']}", help="Sửa"):
                        st.session_state["editing"] = emp["employee_id"]
                with bc2:
                    if st.button("🗑️", key=f"del_{emp['employee_id']}", help="Xóa"):
                        delete_embedding(emp["employee_id"])
                        delete_portraits(emp["employee_id"])
                        delete_employee(emp["employee_id"])
                        st.rerun()

            # Edit form
            if st.session_state.get("editing") == emp["employee_id"]:
                with st.form(f"edit_form_{emp['employee_id']}"):
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        new_name = st.text_input("Họ tên", value=emp.get("full_name", ""))
                        new_dept = st.text_input("Phòng ban", value=emp.get("department", ""))
                    with ec2:
                        new_pos = st.text_input("Chức vụ", value=emp.get("position", ""))
                        new_email = st.text_input("Email", value=emp.get("email", "") or "")

                    if st.form_submit_button("💾 Cập nhật"):
                        update_employee(emp["employee_id"], {
                            "full_name": new_name,
                            "department": new_dept,
                            "position": new_pos,
                            "email": new_email or None,
                        })
                        st.session_state["editing"] = None
                        st.rerun()

            st.divider()

    st.caption(f"Hiển thị {len(employees)} nhân viên")
