# check_dataset_split.py
import os

face_root = "dataset/RWFRD/AFDB_face_dataset/AFDB_face_dataset"
masked_root = "dataset/RWFRD/AFDB_masked_face_dataset/AFDB_masked_face_dataset"

train_txt = "dataset/RWFRD/train_identities.txt"
test_txt = "dataset/RWFRD/test_identities.txt"

face_ids = {d for d in os.listdir(face_root) if os.path.isdir(os.path.join(face_root, d))}
masked_ids = {d for d in os.listdir(masked_root) if os.path.isdir(os.path.join(masked_root, d))}
common_ids = face_ids & masked_ids

train_ids = {x.strip() for x in open(train_txt, encoding="utf-8") if x.strip()}
test_ids = {x.strip() for x in open(test_txt, encoding="utf-8") if x.strip()}

print("Face ids:", len(face_ids))
print("Masked ids:", len(masked_ids))
print("Common ids:", len(common_ids))
print("Train ids:", len(train_ids))
print("Test ids:", len(test_ids))
print("Train ∩ Test:", len(train_ids & test_ids))
print("Train ∩ Common:", len(train_ids & common_ids))
print("Test ∩ Common:", len(test_ids & common_ids))

print("Only train not common sample:", list(train_ids - common_ids)[:20])
print("Only test not common sample:", list(test_ids - common_ids)[:20])