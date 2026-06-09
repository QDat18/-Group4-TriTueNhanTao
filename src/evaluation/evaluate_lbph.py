"""
Script đánh giá hiệu năng phương pháp nhận diện truyền thống LBPH (Local Binary Patterns Histograms) trên tập dữ liệu chuẩn LFW.
So sánh đối chiếu trực tiếp hiệu năng giữa phương pháp Computer Vision truyền thống và Học Sâu (Deep Learning - ArcFace).
"""

import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay


def compute_lbp(img_gray):
    """
    Tính toán ảnh LBP (Local Binary Pattern) sử dụng NumPy slice để tối ưu hóa tốc độ.
    """
    h, w = img_gray.shape
    # Tạo ảnh LBP kích thước h-2, w-2 do bỏ qua viền ngoài cùng 1 pixel
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    
    center = img_gray[1:-1, 1:-1]
    
    # So sánh độ sáng của tâm với 8 điểm lân cận
    n0 = img_gray[0:-2, 0:-2] >= center
    n1 = img_gray[0:-2, 1:-1] >= center
    n2 = img_gray[0:-2, 2:]   >= center
    n3 = img_gray[1:-1, 2:]   >= center
    n4 = img_gray[2:,   2:]   >= center
    n5 = img_gray[2:,   1:-1] >= center
    n6 = img_gray[2:,   0:-2] >= center
    n7 = img_gray[1:-1, 0:-2] >= center
    
    # Tạo mã 8-bit từ kết quả so sánh
    lbp = (n0 << 7) | (n1 << 6) | (n2 << 5) | (n3 << 4) | (n4 << 3) | (n5 << 2) | (n6 << 1) | n7
    return lbp


def extract_lbph_features(image_path, grid_x=8, grid_y=8):
    """
    Trích xuất vector đặc trưng LBPH từ đường dẫn ảnh.
    Chia ảnh thành lưới grid_x x grid_y ô, tính histogram 256 bins cho mỗi ô và nối lại.
    """
    # 1. Đọc ảnh và chuyển sang ảnh xám
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize về kích thước chuẩn 112x112 (tương thích với ảnh input của hệ thống)
    if gray.shape != (112, 112):
        gray = cv2.resize(gray, (112, 112))
        
    # 3. Tính toán ma trận LBP (kích thước 110 x 110)
    lbp = compute_lbp(gray)
    h, w = lbp.shape
    
    # Kích thước mỗi ô lưới
    cell_h = h // grid_y
    cell_w = w // grid_x
    
    hists = []
    # 4. Tính toán histogram cho từng ô lưới
    for i in range(grid_y):
        for j in range(grid_x):
            cell = lbp[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
            hist, _ = np.histogram(cell, bins=256, range=(0, 256))
            
            # Chuẩn hóa L1 histogram của ô lưới
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)
            hists.append(hist)
            
    # Nối tất cả các local histograms thành vector đặc trưng duy nhất (độ dài: 8x8x256 = 16,384)
    feature_vector = np.concatenate(hists)
    return feature_vector


def chi_square_distance(hist1, hist2):
    """
    Tính khoảng cách Chi-bình phương (Chi-Square) giữa hai histogram.
    Đây là khoảng cách tiêu chuẩn tốt nhất để so sánh LBP Histograms.
    """
    # Khoảng cách nhỏ hơn -> Ảnh giống nhau hơn
    return np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))


def read_pairs(pairs_file, root_dir):
    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            label = int(parts[0])
            img1_path = os.path.join(root_dir, parts[1])
            img2_path = os.path.join(root_dir, parts[2])
            pairs.append((img1_path, img2_path, label))
    return pairs


def calculate_best_threshold(labels, scores):
    best_acc = 0.0
    best_threshold = 0.0
    # Chi-square distance thường nằm trong khoảng 0 đến 2.0 (do L1 normalized)
    thresholds = np.linspace(0, 2.0, 1000)

    labels = np.array(labels)
    scores = np.array(scores)

    for threshold in thresholds:
        # Nếu khoảng cách <= threshold -> Nhận định là "Same" (1)
        preds = (scores <= threshold).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold, best_acc


def calculate_eer(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    # Đảo ngược scores vì Chi-Square là khoảng cách (khoảng cách càng nhỏ càng giống)
    # roc_curve yêu cầu scores càng lớn thì khả năng là dương tính càng cao, nên ta truyền vào -scores
    fpr, tpr, thresholds = roc_curve(labels, -scores)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    # Đổi ngược lại ngưỡng EER từ giá trị âm
    eer_threshold = -thresholds[idx]

    return eer, eer_threshold


def main():
    val_root = "dataset/benchmark/val"
    pairs_file = os.path.join(val_root, "lfw_ann.txt")
    output_dir = "outputs/evaluation"
    output_csv = os.path.join(output_dir, "lfw_lbph_eval.csv")
    
    print("[INFO] Reading LFW pairs...")
    pairs = read_pairs(pairs_file, val_root)
    
    labels = []
    distances = []
    missing = 0
    failed = 0
    
    # Cache features to avoid repeating extraction
    features_cache = {}
    
    print("[INFO] Extracting LBPH features and evaluating pairs...")
    for img1, img2, label in tqdm(pairs, desc="Evaluating LBPH"):
        if not os.path.exists(img1) or not os.path.exists(img2):
            missing += 1
            continue
            
        try:
            if img1 not in features_cache:
                features_cache[img1] = extract_lbph_features(img1)
            if img2 not in features_cache:
                features_cache[img2] = extract_lbph_features(img2)
                
            feat1 = features_cache[img1]
            feat2 = features_cache[img2]
            
            # Compute Chi-Square distance
            dist = chi_square_distance(feat1, feat2)
            
            labels.append(label)
            distances.append(dist)
        except Exception as e:
            failed += 1
            print(f"[WARNING] Skip pair due to error: {e}")
            
    if len(labels) == 0:
        print("[ERROR] No valid data to evaluate!")
        return
        
    labels_np = np.array(labels)
    distances_np = np.array(distances)
    
    # 1. Find best threshold and calculate metrics
    best_threshold, accuracy = calculate_best_threshold(labels_np, distances_np)
    auc_score = roc_auc_score(labels_np, -distances_np)
    eer, eer_threshold = calculate_eer(labels_np, distances_np)
    
    preds = (distances_np <= best_threshold).astype(int)
    tp = np.sum((preds == 1) & (labels_np == 1))
    tn = np.sum((preds == 0) & (labels_np == 0))
    fp = np.sum((preds == 1) & (labels_np == 0))
    fn = np.sum((preds == 0) & (labels_np == 1))
    
    far = fp / max(1, fp + tn)
    frr = fn / max(1, fn + tp)
    
    print("\n" + "="*50)
    print("LBPH ON LFW EVALUATION RESULTS:")
    print(f"Total Valid Pairs: {len(labels)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"AUC: {auc_score:.4f}")
    print(f"EER: {eer * 100:.2f}%")
    print(f"Best Distance Threshold: {best_threshold:.4f}")
    print(f"FAR: {far * 100:.2f}% | FRR: {frr * 100:.2f}%")
    print("="*50 + "\n")
    
    # 2. Write CSV results
    os.makedirs(output_dir, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "total_pairs", "valid_pairs", "missing_pairs", "failed_pairs",
            "accuracy", "auc", "eer", "best_threshold", "eer_threshold",
            "far", "frr", "tp", "tn", "fp", "fn"
        ])
        writer.writerow([
            len(pairs), len(labels), missing, failed,
            accuracy, auc_score, eer, best_threshold, eer_threshold,
            far, frr, tp, tn, fp, fn
        ])
        
    # 3. Draw Confusion Matrix
    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
    disp.plot(cmap="Reds", values_format="d")
    plt.title(f"LFW Confusion Matrix (LBPH)")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "lfw_lbph_eval_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 4. Draw ROC curve
    fpr, tpr, _ = roc_curve(labels_np, -distances_np)
    plt.figure()
    plt.plot(fpr, tpr, color="crimson", lw=2, label=f"ROC curve (area = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"LFW ROC Curve (LBPH)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "lfw_lbph_eval_roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    print(f"[SUCCESS] Saved results and plots to: {output_dir}")


if __name__ == "__main__":
    main()
