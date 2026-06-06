"""Settings page - System configuration."""

import streamlit as st


def render():
    st.markdown("# ⚙️ Settings")
    st.markdown('<div class="section-header">Cấu Hình Hệ Thống</div>', unsafe_allow_html=True)

    # ── Recognition Settings ──
    st.markdown("### 🎯 Recognition Settings")
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.45, 0.01)
        st.caption("Ngưỡng similarity để nhận diện thành công")
    with col2:
        cooldown = st.number_input("Cooldown (seconds)", min_value=10, max_value=3600, value=60)
        st.caption("Thời gian chờ giữa 2 lần chấm công cùng NV")

    st.markdown("---")

    # ── Work Schedule ──
    st.markdown("### 🕐 Giờ Làm Việc")
    c1, c2 = st.columns(2)
    with c1:
        st.time_input("Giờ vào", value=None)
    with c2:
        st.time_input("Giờ ra", value=None)

    late_threshold = st.number_input("Cho phép muộn (phút)", min_value=0, max_value=60, value=30)

    st.markdown("---")

    # ── Database Info ──
    st.markdown("### 🗄️ Database")
    st.markdown("""
    <div class="emp-card">
        <p><strong>Provider:</strong> Supabase (PostgreSQL)</p>
        <p><strong>URL:</strong> yxlvatmaiyhapjdguyco.supabase.co</p>
        <p><strong>Tables:</strong> employees, face_embeddings, attendance_logs, devices</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model Info ──
    st.markdown("### 🤖 AI Model")
    st.markdown("""
    <div class="emp-card">
        <p><strong>Architecture:</strong> ArcFace + ResNet</p>
        <p><strong>Embedding Size:</strong> 512</p>
        <p><strong>Checkpoint:</strong> arcface_vggface2_warmup.pth</p>
        <p><strong>Training Data:</strong> VGGFace2 + RMFRD</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("💾 Lưu Cấu Hình", use_container_width=True, type="primary"):
        st.success("✅ Cấu hình đã được lưu!")
