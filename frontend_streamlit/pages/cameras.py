"""Cameras page - Device/Camera management."""

import streamlit as st
from api_client import list_devices, create_device, toggle_device, delete_device


def render():
    st.markdown("# 🎥 Camera Management")
    st.markdown('<div class="section-header">Quản Lý Camera</div>', unsafe_allow_html=True)

    # ── Add Camera ──
    with st.expander("➕ Thêm Camera Mới"):
        with st.form("add_camera"):
            c1, c2, c3 = st.columns(3)
            with c1:
                dev_id = st.text_input("Camera ID *", placeholder="CAM001")
            with c2:
                dev_name = st.text_input("Tên *", placeholder="Camera Tầng 1")
            with c3:
                location = st.text_input("Vị trí", placeholder="Tầng 1 - Sảnh chính")

            if st.form_submit_button("💾 Lưu", use_container_width=True):
                if dev_id and dev_name:
                    result = create_device({
                        "device_id": dev_id,
                        "device_name": dev_name,
                        "location": location,
                        "is_active": True,
                    })
                    if "error" in result:
                        st.error(f"Lỗi: {result['error']}")
                    else:
                        st.success(f"✅ Đã thêm camera {dev_id}")
                        st.rerun()

    # ── Camera List ──
    devices = list_devices()
    if "error" in devices:
        st.error(f"⚠️ {devices['error']}")
        return

    data = devices.get("data", [])

    if not data:
        st.info("Chưa có camera nào. Hãy thêm camera mới.")
        return

    cols = st.columns(min(3, len(data)))
    for i, dev in enumerate(data):
        with cols[i % len(cols)]:
            is_online = dev.get("is_active", False)
            badge = "badge-online" if is_online else "badge-offline"
            status = "Online" if is_online else "Offline"
            icon = "🟢" if is_online else "🔴"

            st.markdown(f'''
            <div class="emp-card">
                <h3>{icon} {dev["device_id"]}</h3>
                <p><strong>{dev.get("device_name", "")}</strong></p>
                <p style="opacity: 0.7;">{dev.get("location", "")}</p>
                <span class="{badge}">{status}</span>
            </div>''', unsafe_allow_html=True)

            bc1, bc2 = st.columns(2)
            with bc1:
                label = "🔴 Tắt" if is_online else "🟢 Bật"
                if st.button(label, key=f"toggle_{dev['device_id']}", use_container_width=True):
                    toggle_device(dev["device_id"])
                    st.rerun()
            with bc2:
                if st.button("🗑️ Xóa", key=f"deld_{dev['device_id']}", use_container_width=True):
                    delete_device(dev["device_id"])
                    st.rerun()

    st.caption(f"Tổng: {len(data)} cameras")
