"""
Thong ke chi tiet tap VGGFace2 sau khi chia Train / Val / Test.
Xuat danh sach file anh ra train.txt, val.txt, test.txt.

Chay:
    python -m src.datasets.stats_vggface2
"""

import os
import sys
import time
from collections import Counter

# Fix Unicode output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src import config
from src.datasets.dataset_vggface2 import VGGFace2Dataset

# Thu muc luu file split txt
SPLITS_DIR = os.path.join(os.path.dirname(config.VGGFACE2_ROOT), "splits")


def count_images_per_identity(dataset: VGGFace2Dataset) -> Counter:
    """Dem so anh moi identity trong dataset."""
    counter = Counter()
    for _, label in dataset.samples:
        counter[label] += 1
    return counter


def export_split_to_txt(
    split_name: str,
    dataset: VGGFace2Dataset,
    output_dir: str,
) -> str:
    """
    Xuat danh sach anh cua 1 split ra file txt.

    Format moi dong:  <duong_dan_anh> <label>
    Tra ve duong dan file txt da luu.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split_name}.txt")

    # Tao idx_to_class de ghi ten identity
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write(f"# VGGFace2 {split_name} split\n")
        f.write(f"# Total: {len(dataset.samples)} images, {dataset.num_classes} identities\n")
        f.write(f"# Config: MAX_CLASSES={config.MAX_CLASSES}, "
                f"MAX_IMAGES_PER_CLASS={config.MAX_IMAGES_PER_CLASS}, "
                f"SPLIT_RATIO={config.SPLIT_RATIO}\n")
        f.write(f"# Format: <image_path> <label> <identity_name>\n")
        f.write(f"#\n")

        for img_path, label in dataset.samples:
            # Chuyen sang forward slash cho nhat quan
            img_path_clean = img_path.replace("\\", "/")
            identity_name = idx_to_class.get(label, f"unknown_{label}")
            f.write(f"{img_path_clean} {label} {identity_name}\n")

    return output_path


def print_split_stats(name: str, dataset: VGGFace2Dataset) -> dict:
    """In thong ke cho 1 split va tra ve dict summary."""
    counter = count_images_per_identity(dataset)

    total_images = len(dataset)
    num_identities_with_images = len(counter)
    num_identities_total = dataset.num_classes

    if len(counter) > 0:
        counts = list(counter.values())
        avg_per_id = sum(counts) / len(counts)
        min_per_id = min(counts)
        max_per_id = max(counts)
        median_per_id = sorted(counts)[len(counts) // 2]
    else:
        avg_per_id = min_per_id = max_per_id = median_per_id = 0

    # Phan bo so anh theo bucket
    buckets = {
        "1-5 imgs": 0,
        "6-10 imgs": 0,
        "11-20 imgs": 0,
        "21-50 imgs": 0,
        "51-100 imgs": 0,
        "101-200 imgs": 0,
        "200+ imgs": 0,
    }
    for count in counter.values():
        if count <= 5:
            buckets["1-5 imgs"] += 1
        elif count <= 10:
            buckets["6-10 imgs"] += 1
        elif count <= 20:
            buckets["11-20 imgs"] += 1
        elif count <= 50:
            buckets["21-50 imgs"] += 1
        elif count <= 100:
            buckets["51-100 imgs"] += 1
        elif count <= 200:
            buckets["101-200 imgs"] += 1
        else:
            buckets["200+ imgs"] += 1

    print(f"\n{'-' * 50}")
    print(f"  [DIR] {name}")
    print(f"{'-' * 50}")
    print(f"  Total images           : {total_images:,}")
    print(f"  Identities (with imgs) : {num_identities_with_images:,} / {num_identities_total:,}")
    print(f"  Avg images/identity    : {avg_per_id:.1f}")
    print(f"  Min images/identity    : {min_per_id}")
    print(f"  Max images/identity    : {max_per_id}")
    print(f"  Median images/identity : {median_per_id}")
    print(f"\n  Distribution (images/identity):")
    for bucket, count in buckets.items():
        if count > 0:
            bar = "#" * min(count // 10, 40) or "|"
            print(f"    {bucket:>12s} : {count:>5,} identities  {bar}")

    return {
        "name": name,
        "total_images": total_images,
        "identities": num_identities_with_images,
        "avg": avg_per_id,
        "min": min_per_id,
        "max": max_per_id,
    }


def main():
    print("=" * 60)
    print("  VGGFACE2 DATASET STATISTICS -- TRAIN / VAL / TEST SPLIT")
    print("=" * 60)

    print(f"\n  Config:")
    print(f"    VGGFACE2_ROOT         = {config.VGGFACE2_ROOT}")
    print(f"    MAX_CLASSES           = {config.MAX_CLASSES}")
    print(f"    MAX_IMAGES_PER_CLASS  = {config.MAX_IMAGES_PER_CLASS}")
    print(f"    SPLIT_RATIO           = {config.SPLIT_RATIO}")

    max_classes = config.MAX_CLASSES if config.USE_SUBSET else None
    max_images = config.MAX_IMAGES_PER_CLASS if config.USE_SUBSET else None
    split_ratio = config.SPLIT_RATIO

    summaries = []
    exported_files = []

    for split_name in ["train", "val", "test"]:
        t0 = time.time()

        dataset = VGGFace2Dataset(
            root_dir=config.VGGFACE2_ROOT,
            max_classes=max_classes,
            max_images_per_class=max_images,
            split=split_name,
            split_ratio=split_ratio,
        )

        elapsed = time.time() - t0
        stats = print_split_stats(f"{split_name.upper()} SET", dataset)
        stats["load_time"] = elapsed
        summaries.append(stats)

        # Export to txt
        txt_path = export_split_to_txt(split_name, dataset, SPLITS_DIR)
        exported_files.append(txt_path)
        print(f"  >> Exported: {txt_path}")

    # Tong ket
    total = sum(s["total_images"] for s in summaries)
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  {'Split':<12s} {'Images':>10s} {'Ratio':>10s} {'Identities':>12s} {'Avg/ID':>10s}")
    print(f"  {'-' * 56}")

    for s in summaries:
        pct = s["total_images"] / total * 100 if total > 0 else 0
        print(f"  {s['name']:<12s} {s['total_images']:>10,} {pct:>9.1f}% {s['identities']:>12,} {s['avg']:>10.1f}")

    print(f"  {'-' * 56}")
    print(f"  {'TOTAL':<12s} {total:>10,} {'100.0%':>10s}")

    print(f"\n  Load time:")
    for s in summaries:
        print(f"    {s['name']:<12s}: {s['load_time']:.2f}s")

    print(f"\n  Exported split files:")
    for f in exported_files:
        size_kb = os.path.getsize(f) / 1024
        print(f"    {f}  ({size_kb:.1f} KB)")

    print()


if __name__ == "__main__":
    main()

