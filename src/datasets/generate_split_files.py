import os

AFDB_FACE_ROOT = "dataset/RWFRD/AFDB_face_dataset/AFDB_face_dataset"
AFDB_MASKED_ROOT = "dataset/RWFRD/AFDB_masked_face_dataset/AFDB_masked_face_dataset"
OUTPUT_DIR = "dataset/RWFRD"
SPLIT_RATIO = 0.8


def main():
    print("=" * 70)
    print("GENERATING TRAIN/TEST IDENTITY SPLIT FILES")
    print("=" * 70)

    # 1. Scan all unique identity folder names across both roots
    identity_names = set()

    for root in [AFDB_FACE_ROOT, AFDB_MASKED_ROOT]:
        if not os.path.exists(root):
            print(f"[WARNING] Directory not found: {root}")
            continue

        for folder in os.listdir(root):
            if os.path.isdir(os.path.join(root, folder)):
                identity_names.add(folder)

    identity_names = sorted(list(identity_names))
    total_identities = len(identity_names)

    print(f"Total unique identities found: {total_identities}")

    if total_identities == 0:
        print("[ERROR] No identities found! Please check your dataset path.")
        return

    # 2. Split into train and test deterministically
    split_idx = int(total_identities * SPLIT_RATIO)
    train_ids = identity_names[:split_idx]
    test_ids = identity_names[split_idx:]

    print(f"Train identities (80%): {len(train_ids)}")
    print(f"Test identities (20%): {len(test_ids)}")

    # 3. Write to .txt files
    train_txt_path = os.path.join(OUTPUT_DIR, "train_identities.txt")
    test_txt_path = os.path.join(OUTPUT_DIR, "test_identities.txt")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(train_txt_path, "w", encoding="utf-8") as f:
        for identity in train_ids:
            f.write(identity + "\n")

    with open(test_txt_path, "w", encoding="utf-8") as f:
        for identity in test_ids:
            f.write(identity + "\n")

    print(f"Saved train split to: {train_txt_path}")
    print(f"Saved test split to: {test_txt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
