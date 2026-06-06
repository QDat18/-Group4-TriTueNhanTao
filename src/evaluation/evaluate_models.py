import os
import csv
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from src.models.face_recognition_model import (
    FaceRecognitionModel
)

from src.utils.transforms import (
    get_val_transform
)

from src.attendance.attendance_service import (
    AttendanceService
)

from src.config import (
    RECOGNITION_THRESHOLD
)


TEST_ROOT = "dataset/test"

REPORT_DIR = "evaluation_reports"


class ModelEvaluator:
    """
    Đánh giá toàn diện hệ thống nhận diện khuôn mặt.

    Bao gồm:
      - Accuracy, Precision, Recall, F1 (tổng thể + từng người)
      - False Accept Rate (FAR) / False Reject Rate (FRR)
      - Phân bố điểm similarity (genuine vs impostor)
      - Phân tích ngưỡng (threshold sweep) và Equal Error Rate (EER)
      - Rank-K Identification Rate
      - Confusion Matrix
      - Xuất báo cáo CSV + biểu đồ PNG
    """

    def __init__(self):

        self.model = FaceRecognitionModel()

        self.service = AttendanceService()

        self.transform = get_val_transform()

        # ── Bộ đếm tổng thể ──

        self.total = 0
        self.correct = 0

        self.false_accept = 0
        self.false_reject = 0

        # ── Lưu chi tiết từng mẫu ──

        self.records = []

        # ── Lưu per-identity ──

        self.per_identity = defaultdict(
            lambda: {
                "total": 0,
                "correct": 0,
                "false_accept": 0,
                "false_reject": 0,
                "similarities": [],
            }
        )

        # ── Phân bố similarity ──

        self.genuine_scores = []
        self.impostor_scores = []

        # ── Timestamp ──

        self.timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        os.makedirs(REPORT_DIR, exist_ok=True)

    # ════════════════════════════════════════
    # EMBEDDING
    # ════════════════════════════════════════

    def get_embedding(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        image = image.unsqueeze(0)

        embedding = (
            self.model
            .get_embedding(image)
        )

        return (
            embedding
            .squeeze(0)
            .numpy()
        )

    # ════════════════════════════════════════
    # RANK-K: tính similarity với tất cả
    # ════════════════════════════════════════

    def get_all_similarities(self, embedding):
        """
        Trả về danh sách (employee_id, similarity)
        sắp xếp giảm dần theo similarity.
        """

        results = []

        for emp_id, data in (
            self.service
            .employee_embeddings
            .items()
        ):

            score = (
                self.service
                .cosine_similarity(
                    embedding,
                    data["embedding"]
                )
            )

            results.append(
                (emp_id, score)
            )

        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return results

    # ════════════════════════════════════════
    # VÒNG LẶP ĐÁNH GIÁ CHÍNH
    # ════════════════════════════════════════

    def evaluate(self):

        print()
        print("=" * 60)
        print(
            "  ĐÁNH GIÁ TOÀN DIỆN HỆ THỐNG"
            " NHẬN DIỆN KHUÔN MẶT"
        )
        print("=" * 60)
        print(
            f"  Threshold : {RECOGNITION_THRESHOLD}"
        )
        print(
            f"  Test Root : {TEST_ROOT}"
        )
        print("=" * 60)
        print()

        start_time = time.time()

        identities = sorted(
            os.listdir(TEST_ROOT)
        )

        for employee_id in identities:

            employee_dir = os.path.join(
                TEST_ROOT,
                employee_id
            )

            if not os.path.isdir(
                employee_dir
            ):
                continue

            images = [
                f for f in os.listdir(
                    employee_dir
                )
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            for image_name in tqdm(
                images,
                desc=f"[{employee_id}]"
            ):

                image_path = os.path.join(
                    employee_dir,
                    image_name
                )

                embedding = self.get_embedding(
                    image_path
                )

                # ── Tính similarity với tất cả ──

                all_sims = (
                    self.get_all_similarities(
                        embedding
                    )
                )

                # ── Kết quả best match ──

                result = (
                    self.service
                    .find_best_match(
                        embedding
                    )
                )

                self.total += 1

                id_stats = (
                    self.per_identity[
                        employee_id
                    ]
                )

                id_stats["total"] += 1

                if result is None:

                    # Không tìm thấy ai → FRR

                    self.false_reject += 1
                    id_stats["false_reject"] += 1

                    best_score = (
                        all_sims[0][1]
                        if all_sims
                        else 0.0
                    )

                    self.records.append({
                        "image": image_path,
                        "true_id": employee_id,
                        "predicted_id": "REJECTED",
                        "similarity": best_score,
                        "rank": -1,
                        "verdict": "FALSE_REJECT",
                    })

                else:

                    predicted = result[
                        "employee_id"
                    ]

                    similarity = result[
                        "similarity"
                    ]

                    # ── Tính rank ──

                    rank = -1

                    for idx, (eid, _) in enumerate(
                        all_sims
                    ):
                        if eid == employee_id:
                            rank = idx + 1
                            break

                    if predicted == employee_id:

                        self.correct += 1
                        id_stats["correct"] += 1

                        id_stats[
                            "similarities"
                        ].append(similarity)

                        self.genuine_scores.append(
                            similarity
                        )

                        self.records.append({
                            "image": image_path,
                            "true_id": employee_id,
                            "predicted_id": predicted,
                            "similarity": similarity,
                            "rank": rank,
                            "verdict": "CORRECT",
                        })

                    else:

                        self.false_accept += 1
                        id_stats["false_accept"] += 1

                        self.impostor_scores.append(
                            similarity
                        )

                        self.records.append({
                            "image": image_path,
                            "true_id": employee_id,
                            "predicted_id": predicted,
                            "similarity": similarity,
                            "rank": rank,
                            "verdict": "FALSE_ACCEPT",
                        })

                # ── Tính impostor scores bổ sung ──
                # Lấy score cao nhất với người SAI

                for eid, sc in all_sims:

                    if eid != employee_id:
                        self.impostor_scores.append(
                            sc
                        )
                        break

        elapsed = time.time() - start_time

        # ── Xuất tất cả báo cáo ──

        self.report_summary(elapsed)
        self.report_per_identity()
        self.report_rank_k()
        self.report_threshold_analysis()
        self.export_csv()
        self.plot_similarity_distribution()
        self.plot_threshold_curve()
        self.plot_confusion_matrix()

        print()
        print("=" * 60)
        print(
            f"  Báo cáo đã lưu vào: {REPORT_DIR}/"
        )
        print("=" * 60)

    # ════════════════════════════════════════
    # 1. BÁO CÁO TỔNG QUÁT
    # ════════════════════════════════════════

    def report_summary(self, elapsed):

        accuracy = (
            self.correct
            / max(1, self.total)
        )

        far = (
            self.false_accept
            / max(1, self.total)
        )

        frr = (
            self.false_reject
            / max(1, self.total)
        )

        precision = (
            self.correct
            / max(
                1,
                self.correct
                + self.false_accept
            )
        )

        recall = (
            self.correct
            / max(
                1,
                self.correct
                + self.false_reject
            )
        )

        f1 = (
            2 * precision * recall
            / max(
                1e-9,
                precision + recall
            )
        )

        print()
        print("=" * 60)
        print("  1. KẾT QUẢ TỔNG QUÁT")
        print("=" * 60)

        print(
            f"  Tổng mẫu test    : {self.total}"
        )

        print(
            f"  Nhận đúng        : {self.correct}"
        )

        print(
            f"  Nhận nhầm (FA)   : {self.false_accept}"
        )

        print(
            f"  Từ chối nhầm (FR): {self.false_reject}"
        )

        print("-" * 60)

        print(
            f"  Accuracy   : {accuracy * 100:.2f}%"
        )

        print(
            f"  Precision  : {precision * 100:.2f}%"
        )

        print(
            f"  Recall     : {recall * 100:.2f}%"
        )

        print(
            f"  F1 Score   : {f1 * 100:.2f}%"
        )

        print("-" * 60)

        print(
            f"  FAR        : {far * 100:.2f}%"
        )

        print(
            f"  FRR        : {frr * 100:.2f}%"
        )

        print(
            f"  Thời gian  : {elapsed:.1f}s"
        )

        print(
            f"  Threshold  : {RECOGNITION_THRESHOLD}"
        )

        print("=" * 60)

    # ════════════════════════════════════════
    # 2. BÁO CÁO TỪNG NGƯỜI
    # ════════════════════════════════════════

    def report_per_identity(self):

        print()
        print("=" * 60)
        print("  2. CHI TIẾT TỪNG NHÂN VIÊN")
        print("=" * 60)

        header = (
            f"  {'ID':<15}"
            f"{'Total':>6}"
            f"{'Correct':>9}"
            f"{'FA':>5}"
            f"{'FR':>5}"
            f"{'Acc%':>8}"
            f"{'AvgSim':>8}"
        )

        print(header)
        print("-" * 60)

        for emp_id in sorted(
            self.per_identity.keys()
        ):

            stats = self.per_identity[emp_id]

            total = stats["total"]
            correct = stats["correct"]
            fa = stats["false_accept"]
            fr = stats["false_reject"]

            acc = (
                correct
                / max(1, total)
                * 100
            )

            sims = stats["similarities"]

            avg_sim = (
                np.mean(sims)
                if sims
                else 0.0
            )

            print(
                f"  {emp_id:<15}"
                f"{total:>6}"
                f"{correct:>9}"
                f"{fa:>5}"
                f"{fr:>5}"
                f"{acc:>7.1f}%"
                f"{avg_sim:>8.4f}"
            )

        print("=" * 60)

    # ════════════════════════════════════════
    # 3. RANK-K IDENTIFICATION RATE
    # ════════════════════════════════════════

    def report_rank_k(self):

        ranks = [
            r["rank"]
            for r in self.records
            if r["rank"] > 0
        ]

        if not ranks:
            print(
                "\n  Không có dữ liệu"
                " rank để tính."
            )
            return

        print()
        print("=" * 60)
        print("  3. RANK-K IDENTIFICATION RATE")
        print("=" * 60)

        total_with_rank = len(ranks)

        for k in [1, 3, 5, 10]:

            count = sum(
                1 for r in ranks if r <= k
            )

            rate = (
                count
                / total_with_rank
                * 100
            )

            print(
                f"  Rank-{k:<3}:"
                f" {rate:>6.2f}%"
                f"  ({count}/{total_with_rank})"
            )

        print("=" * 60)

    # ════════════════════════════════════════
    # 4. PHÂN TÍCH NGƯỠNG + EER
    # ════════════════════════════════════════

    def report_threshold_analysis(self):

        genuine = np.array(
            self.genuine_scores
        )

        impostor = np.array(
            self.impostor_scores
        )

        if len(genuine) == 0 or len(impostor) == 0:
            print(
                "\n  Không đủ dữ liệu"
                " để phân tích ngưỡng."
            )
            return

        print()
        print("=" * 60)
        print(
            "  4. PHÂN TÍCH NGƯỠNG"
            " (THRESHOLD SWEEP)"
        )
        print("=" * 60)

        thresholds = np.arange(
            0.0, 1.01, 0.05
        )

        header = (
            f"  {'Threshold':>10}"
            f"{'FAR%':>8}"
            f"{'FRR%':>8}"
            f"{'Acc%':>8}"
        )

        print(header)
        print("-" * 60)

        best_eer_diff = float("inf")
        eer_threshold = 0.0
        eer_value = 0.0

        for thr in thresholds:

            # FRR: genuine bị reject (< threshold)
            frr = (
                np.sum(genuine < thr)
                / max(1, len(genuine))
            )

            # FAR: impostor được accept (>= threshold)
            far = (
                np.sum(impostor >= thr)
                / max(1, len(impostor))
            )

            acc = 1.0 - (far + frr) / 2.0

            print(
                f"  {thr:>10.2f}"
                f"{far * 100:>8.2f}"
                f"{frr * 100:>8.2f}"
                f"{acc * 100:>8.2f}"
            )

            diff = abs(far - frr)

            if diff < best_eer_diff:
                best_eer_diff = diff
                eer_threshold = thr
                eer_value = (far + frr) / 2.0

        print("-" * 60)

        print(
            f"  Equal Error Rate (EER)"
            f" ≈ {eer_value * 100:.2f}%"
            f" tại threshold"
            f" = {eer_threshold:.2f}"
        )

        print("=" * 60)

        # ── Thống kê similarity ──

        print()
        print("=" * 60)
        print(
            "  THỐNG KÊ PHÂN BỐ SIMILARITY"
        )
        print("=" * 60)

        print(
            f"  Genuine  — "
            f"Mean: {np.mean(genuine):.4f}  "
            f"Std: {np.std(genuine):.4f}  "
            f"Min: {np.min(genuine):.4f}  "
            f"Max: {np.max(genuine):.4f}"
        )

        print(
            f"  Impostor — "
            f"Mean: {np.mean(impostor):.4f}  "
            f"Std: {np.std(impostor):.4f}  "
            f"Min: {np.min(impostor):.4f}  "
            f"Max: {np.max(impostor):.4f}"
        )

        print("=" * 60)

    # ════════════════════════════════════════
    # 5. XUẤT CSV
    # ════════════════════════════════════════

    def export_csv(self):

        # ── Chi tiết từng mẫu ──

        detail_path = os.path.join(
            REPORT_DIR,
            f"detail_{self.timestamp}.csv"
        )

        with open(
            detail_path, "w",
            newline="", encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "true_id",
                    "predicted_id",
                    "similarity",
                    "rank",
                    "verdict",
                ]
            )

            writer.writeheader()
            writer.writerows(self.records)

        print(
            f"\n  CSV chi tiết: {detail_path}"
        )

        # ── Per-identity ──

        identity_path = os.path.join(
            REPORT_DIR,
            f"per_identity_{self.timestamp}.csv"
        )

        with open(
            identity_path, "w",
            newline="", encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "employee_id",
                "total",
                "correct",
                "false_accept",
                "false_reject",
                "accuracy",
                "avg_similarity",
            ])

            for emp_id in sorted(
                self.per_identity.keys()
            ):

                s = self.per_identity[emp_id]

                acc = (
                    s["correct"]
                    / max(1, s["total"])
                )

                sims = s["similarities"]

                avg = (
                    float(np.mean(sims))
                    if sims
                    else 0.0
                )

                writer.writerow([
                    emp_id,
                    s["total"],
                    s["correct"],
                    s["false_accept"],
                    s["false_reject"],
                    f"{acc:.4f}",
                    f"{avg:.4f}",
                ])

        print(
            f"  CSV per-identity: {identity_path}"
        )

    # ════════════════════════════════════════
    # 6. BIỂU ĐỒ PHÂN BỐ SIMILARITY
    # ════════════════════════════════════════

    def plot_similarity_distribution(self):

        genuine = np.array(
            self.genuine_scores
        )

        impostor = np.array(
            self.impostor_scores
        )

        if len(genuine) == 0 and len(impostor) == 0:
            return

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        if len(genuine) > 0:
            ax.hist(
                genuine,
                bins=50,
                alpha=0.6,
                color="#2ecc71",
                label="Genuine (đúng người)",
                density=True,
            )

        if len(impostor) > 0:
            ax.hist(
                impostor,
                bins=50,
                alpha=0.6,
                color="#e74c3c",
                label="Impostor (sai người)",
                density=True,
            )

        ax.axvline(
            x=RECOGNITION_THRESHOLD,
            color="#f39c12",
            linestyle="--",
            linewidth=2,
            label=(
                f"Threshold ="
                f" {RECOGNITION_THRESHOLD}"
            ),
        )

        ax.set_xlabel(
            "Cosine Similarity",
            fontsize=12
        )

        ax.set_ylabel(
            "Density",
            fontsize=12
        )

        ax.set_title(
            "Phân bố Similarity:"
            " Genuine vs Impostor",
            fontsize=14,
            fontweight="bold",
        )

        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        path = os.path.join(
            REPORT_DIR,
            f"similarity_dist_{self.timestamp}.png"
        )

        fig.savefig(path, dpi=150)
        plt.close(fig)

        print(
            f"  Biểu đồ similarity: {path}"
        )

    # ════════════════════════════════════════
    # 7. BIỂU ĐỒ FAR / FRR THEO THRESHOLD
    # ════════════════════════════════════════

    def plot_threshold_curve(self):

        genuine = np.array(
            self.genuine_scores
        )

        impostor = np.array(
            self.impostor_scores
        )

        if len(genuine) == 0 or len(impostor) == 0:
            return

        thresholds = np.arange(
            0.0, 1.001, 0.01
        )

        fars = []
        frrs = []

        for thr in thresholds:

            frr = (
                np.sum(genuine < thr)
                / max(1, len(genuine))
            )

            far = (
                np.sum(impostor >= thr)
                / max(1, len(impostor))
            )

            fars.append(far)
            frrs.append(frr)

        fars = np.array(fars)
        frrs = np.array(frrs)

        # ── Tìm EER ──

        eer_idx = np.argmin(
            np.abs(fars - frrs)
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.plot(
            thresholds, fars * 100,
            color="#e74c3c",
            linewidth=2,
            label="FAR (False Accept Rate)",
        )

        ax.plot(
            thresholds, frrs * 100,
            color="#3498db",
            linewidth=2,
            label="FRR (False Reject Rate)",
        )

        ax.axvline(
            x=RECOGNITION_THRESHOLD,
            color="#f39c12",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Current Threshold"
                f" = {RECOGNITION_THRESHOLD}"
            ),
        )

        ax.plot(
            thresholds[eer_idx],
            fars[eer_idx] * 100,
            "ko",
            markersize=8,
            label=(
                f"EER ≈"
                f" {fars[eer_idx] * 100:.2f}%"
                f" @ {thresholds[eer_idx]:.2f}"
            ),
        )

        ax.set_xlabel(
            "Threshold",
            fontsize=12
        )

        ax.set_ylabel(
            "Error Rate (%)",
            fontsize=12
        )

        ax.set_title(
            "FAR / FRR theo Threshold",
            fontsize=14,
            fontweight="bold",
        )

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        path = os.path.join(
            REPORT_DIR,
            f"threshold_curve_{self.timestamp}.png"
        )

        fig.savefig(path, dpi=150)
        plt.close(fig)

        print(
            f"  Biểu đồ threshold: {path}"
        )

    # ════════════════════════════════════════
    # 8. CONFUSION MATRIX
    # ════════════════════════════════════════

    def plot_confusion_matrix(self):

        labels = sorted(
            self.per_identity.keys()
        )

        if not labels:
            return

        # Thêm nhãn "REJECTED"
        all_labels = labels + ["REJECTED"]

        label_to_idx = {
            lbl: idx
            for idx, lbl in enumerate(
                all_labels
            )
        }

        n = len(all_labels)

        matrix = np.zeros(
            (n, n),
            dtype=int
        )

        for rec in self.records:

            true = rec["true_id"]
            pred = rec["predicted_id"]

            if true not in label_to_idx:
                continue

            if pred not in label_to_idx:
                continue

            i = label_to_idx[true]
            j = label_to_idx[pred]

            matrix[i][j] += 1

        # ── Vẽ ──

        fig_size = max(8, n * 0.6)

        fig, ax = plt.subplots(
            figsize=(fig_size, fig_size)
        )

        im = ax.imshow(
            matrix,
            cmap="Blues",
            interpolation="nearest",
        )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

        ax.set_xticklabels(
            all_labels,
            rotation=45,
            ha="right",
            fontsize=8,
        )

        ax.set_yticklabels(
            all_labels,
            fontsize=8,
        )

        # Hiển thị số trong ô (nếu matrix nhỏ)

        if n <= 20:

            for i in range(n):
                for j in range(n):

                    val = matrix[i][j]

                    if val == 0:
                        continue

                    color = (
                        "white"
                        if val > matrix.max() / 2
                        else "black"
                    )

                    ax.text(
                        j, i,
                        str(val),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=7,
                    )

        ax.set_xlabel(
            "Predicted",
            fontsize=12
        )

        ax.set_ylabel(
            "True",
            fontsize=12
        )

        ax.set_title(
            "Confusion Matrix",
            fontsize=14,
            fontweight="bold",
        )

        fig.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()

        path = os.path.join(
            REPORT_DIR,
            f"confusion_matrix_{self.timestamp}.png"
        )

        fig.savefig(path, dpi=150)
        plt.close(fig)

        print(
            f"  Confusion matrix: {path}"
        )


# ════════════════════════════════════════
# MAIN
# ════════════════════════════════════════


if __name__ == "__main__":

    evaluator = ModelEvaluator()

    evaluator.evaluate()