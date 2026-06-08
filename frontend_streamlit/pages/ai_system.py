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
            with st.spinner("Đang tính toán và cập nhật lại toàn bộ Embeddings lên Supabase..."):
                res = rebuild_embeddings()
                if "error" in res:
                    st.error(f"Lỗi: {res['error']}")
                else:
                    st.success("Đã rebuild và đồng bộ embeddings thành công!")
                    st.rerun()

    # ── Tab 2: Model Evaluation ──
    with tab2:
        st.markdown('<div class="section-header">Model Evaluation</div>', unsafe_allow_html=True)

        import os
        from src.config import FINAL_CHECKPOINT_PATH

        model_filename = os.path.basename(FINAL_CHECKPOINT_PATH)

        st.markdown(f"""
        <div class="emp-card">
            <p style="opacity: 0.7; margin-bottom: 4px;">Mô hình hiện tại (Active Backbone)</p>
            <h3 style="color: #3b82f6; margin-top: 0;">{model_filename}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Choose metrics based on model
        if "finetuned" in model_filename or "rmfrd" in model_filename:
            r1, r5, eer, thr = "7.11%", "23.35%", "32.28%", "0.44"
        elif "warmup" in model_filename:
            r1, r5, eer, thr = "4.57%", "16.24%", "42.80%", "0.11"
        else:
            r1, r5, eer, thr = "—", "—", "—", "—"

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Rank-1 Accuracy", r1)
        with mc2:
            st.metric("Rank-5 Accuracy", r5)
        with mc3:
            st.metric("Equal Error Rate (EER)", eer)
        with mc4:
            st.metric("Ngưỡng tối ưu EER", thr)

        st.markdown("---")

        # Load and render detailed report if exists
        report_path = "evaluation_reports/afdb_masked_evaluation_report.md"
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
            st.markdown("### 📋 Báo cáo đánh giá chi tiết (AFDB Masked Face)")
            st.markdown(report_content)
        else:
            st.info("Chưa có báo cáo đánh giá chi tiết trong thư mục `evaluation_reports/`.")

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
