"""AI System page - Embeddings, Model Evaluation, Anti-Spoofing."""

import streamlit as st
from api_client import list_embeddings, delete_embedding


def render():
    st.markdown("# 🧠 AI System")

    tab1, tab2, tab3 = st.tabs(["📦 Embeddings", "📊 Model Evaluation", "🛡️ Anti-Spoofing"])

    # ── Tab 1: Embeddings ──
    with tab1:
        st.markdown('<div class="section-header">Embedding Database</div>', unsafe_allow_html=True)

        embs = list_embeddings()
        if "error" in embs:
            st.error(f"⚠️ {embs['error']}")
            return

        data = embs.get("data", [])

        if not data:
            st.info("Chưa có embedding nào.")
        else:
            for emb in data:
                c1, c2, c3, c4, c5 = st.columns([2, 2, 1.5, 2, 2])
                with c1:
                    st.markdown(f"**{emb['employee_id']}**")
                with c2:
                    st.markdown(emb.get("full_name", "—"))
                with c3:
                    st.markdown(f"📷 {emb.get('image_count', 0)} ảnh")
                with c4:
                    updated = emb.get("updated_at", "—")
                    if updated and updated != "—":
                        updated = updated[:10]
                    st.markdown(f"🕐 {updated}")
                with c5:
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("🔄", key=f"rebuild_{emb['employee_id']}", help="Rebuild"):
                            st.info("Chạy: `python -m src.attendance.build_embeddings`")
                    with bc2:
                        if st.button("🗑️", key=f"delemb_{emb['employee_id']}", help="Delete"):
                            delete_embedding(emb["employee_id"])
                            st.rerun()
                st.divider()

        st.caption(f"Tổng: {len(data)} embeddings | Embedding size: 512")

        if st.button("🔄 Rebuild All Embeddings", use_container_width=True):
            st.info("💡 Chạy: `python -m src.attendance.build_embeddings`")

    # ── Tab 2: Model Evaluation ──
    with tab2:
        st.markdown('<div class="section-header">Model Evaluation</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="emp-card">
            <p style="opacity: 0.7;">Current Model</p>
            <h3 style="color: #a78bfa;">arcface_vggface2_warmup.pth</h3>
        </div>
        """, unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Accuracy", "—")
        with mc2:
            st.metric("FAR", "—")
        with mc3:
            st.metric("FRR", "—")
        with mc4:
            st.metric("F1 Score", "—")

        st.markdown("---")

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("▶️ Run Evaluation", use_container_width=True, type="primary"):
                st.info("💡 Chạy: `python -m src.evaluation.evaluate_models`")
        with bc2:
            if st.button("📥 Export Report", use_container_width=True):
                st.info("Xem trong thư mục `evaluation_reports/`")

        st.markdown("---")
        st.markdown("""
        > **Metrics bao gồm:**
        > - Accuracy, Precision, Recall, F1
        > - FAR (False Accept Rate), FRR (False Reject Rate)
        > - EER (Equal Error Rate)
        > - Rank-K Identification Rate
        > - Confusion Matrix
        """)

    # ── Tab 3: Anti-Spoofing ──
    with tab3:
        st.markdown('<div class="section-header">Anti-Spoofing System</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="emp-card" style="text-align: center;">
            <p style="font-size: 2rem;">🛡️</p>
            <h3>Liveness Detection</h3>
            <p style="opacity: 0.7;">Phát hiện giả mạo khuôn mặt (ảnh, video, mask)</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("Module anti-spoofing đang trong quá trình phát triển.")
