"""AI System page - Embeddings, Model Evaluation, Anti-Spoofing."""

import streamlit as st
from api_client import list_embeddings, delete_embedding, rebuild_embeddings, reload_embeddings, get_model_info


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
                        # Rebuild individual embedding
                        if st.button("🔄", key=f"rebuild_{emb['employee_id']}", help="Reload embedding into memory"):
                            res = reload_embeddings()
                            if "error" in res:
                                st.error(res["error"])
                            else:
                                st.success("Loaded!")
                                st.rerun()
                    with bc2:
                        if st.button("🗑️", key=f"delemb_{emb['employee_id']}", help="Delete embedding"):
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

        model_info = get_model_info()
        if "error" in model_info:
            st.error(f"⚠️ Không lấy được thông tin mô hình: {model_info['error']}")
        else:
            model_name = model_info.get("model_name", "Unknown Model")
            metrics = model_info.get("metrics", {})

            st.markdown(f"""
            <div class="emp-card">
                <p style="opacity: 0.7; margin-bottom: 4px;">Mô hình hiện tại (Active Backbone)</p>
                <h3 style="color: #3b82f6; margin-top: 0;">{model_name}</h3>
            </div>
            """, unsafe_allow_html=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Rank-1 Accuracy", metrics.get("rank1", "—"))
            with mc2:
                st.metric("Rank-5 Accuracy", metrics.get("rank5", "—"))
            with mc3:
                st.metric("Equal Error Rate (EER)", metrics.get("eer", "—"))
            with mc4:
                st.metric("Ngưỡng tối ưu EER", metrics.get("threshold", "—"))

            st.markdown("---")

            # ── Standard Benchmarks ──
            benchmarks = model_info.get("benchmarks", [])
            if benchmarks:
                import pandas as pd
                df_bench = pd.DataFrame(benchmarks)
                col_map = {
                    "dataset": "Dataset",
                    "pairs": "Số cặp (Pairs)",
                    "accuracy": "Độ chính xác (Accuracy)",
                    "auc": "AUC",
                    "eer": "EER",
                    "threshold": "Ngưỡng tối ưu (Threshold)",
                    "far": "FAR (Nhận nhầm)",
                    "frr": "FRR (Từ chối sai)"
                }
                df_bench_display = df_bench.rename(columns=col_map)
                st.markdown("#### 📊 Kết quả kiểm thử trên các tập dữ liệu tiêu chuẩn (Benchmarks)")
                st.dataframe(df_bench_display, use_container_width=True, hide_index=True)
                
                # ── Plots Analysis ──
                st.markdown("#### 📈 Biểu đồ Đánh giá Chi tiết (ROC & Confusion Matrix)")
                import os
                selected_ds = st.selectbox("Chọn tập dữ liệu hiển thị:", ["LFW", "CALFW", "CPLFW", "AGEDB"])
                ds_filename_map = {
                    "LFW": "lfw_eval",
                    "CALFW": "calfw_eval",
                    "CPLFW": "cplfw_eval",
                    "AGEDB": "agedb30_eval"
                }
                prefix = ds_filename_map[selected_ds]
                cm_img_path = os.path.join("outputs", "evaluation", f"{prefix}_confusion_matrix.png")
                roc_img_path = os.path.join("outputs", "evaluation", f"{prefix}_roc_curve.png")
                
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.markdown("<p style='text-align: center; font-weight: bold;'>Ma trận nhầm lẫn (Confusion Matrix)</p>", unsafe_allow_html=True)
                    if os.path.exists(cm_img_path):
                        st.image(cm_img_path, use_column_width=True)
                    else:
                        st.info("Chưa có ảnh ma trận nhầm lẫn cho tập này.")
                with pc2:
                    st.markdown("<p style='text-align: center; font-weight: bold;'>Đường cong ROC (ROC Curve)</p>", unsafe_allow_html=True)
                    if os.path.exists(roc_img_path):
                        st.image(roc_img_path, use_column_width=True)
                    else:
                        st.info("Chưa có ảnh đường cong ROC cho tập này.")
            else:
                st.warning("Chưa có dữ liệu đánh giá benchmark.")

            st.markdown("---")

            # Load and render detailed report if exists
            report_content = model_info.get("report", "")
            if report_content:
                st.markdown("### 📋 Báo cáo đánh giá chi tiết (AFDB Masked Face)")
                st.markdown(report_content)
            else:
                st.info("Chưa có báo cáo đánh giá chi tiết từ API backend.")

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

        st.info("Module anti-spoofing tích hợp trực tiếp trong Realtime Attendance.")
