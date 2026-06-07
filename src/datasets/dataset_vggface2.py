import os
import random
from typing import Callable, Optional, Tuple, List, Dict

from PIL import Image
from torch.utils.data import Dataset


class VGGFace2Dataset(Dataset):
    """
    VGGFace2 Dataset với hỗ trợ chia train/val/test.

    Chia ảnh trong mỗi identity theo tỷ lệ (mặc định 80/10/10).
    Tất cả các split dùng chung class_to_idx mapping để ArcFace head
    có thể dùng chung num_classes.

    Args:
        root_dir: Đường dẫn thư mục gốc chứa các folder identity.
        transform: Transform áp dụng cho ảnh.
        max_classes: Số identity tối đa (subset).
        max_images_per_class: Số ảnh tối đa mỗi identity.
        split: "train", "val", "test", hoặc None (dùng toàn bộ).
        split_ratio: Tuple (train, val, test), tổng = 1.0.
        seed: Seed cố định để chia reproducible.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        max_classes: Optional[int] = None,
        max_images_per_class: Optional[int] = None,
        split: Optional[str] = None,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.max_classes = max_classes
        self.max_images_per_class = max_images_per_class
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed

        assert split in (None, "train", "val", "test"), \
            f"split phải là None, 'train', 'val' hoặc 'test', nhận được: {split}"

        if split is not None:
            assert abs(sum(split_ratio) - 1.0) < 1e-6, \
                f"split_ratio phải có tổng = 1.0, nhận được: {sum(split_ratio)}"

        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.num_classes: int = 0

        self._load_samples()

    def _load_samples(self) -> None:
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {self.root_dir}")

        identities = sorted([
            folder for folder in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, folder))
        ])

        if self.max_classes is not None:
            identities = identities[:self.max_classes]

        # Xây dựng class_to_idx cho TẤT CẢ identities (dùng chung giữa các split)
        for label, identity in enumerate(identities):
            self.class_to_idx[identity] = label

        self.num_classes = len(self.class_to_idx)

        rng = random.Random(self.seed)

        for identity in identities:
            label = self.class_to_idx[identity]
            identity_dir = os.path.join(self.root_dir, identity)

            image_files = sorted([
                f for f in os.listdir(identity_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            if self.max_images_per_class is not None:
                image_files = image_files[:self.max_images_per_class]

            if self.split is not None and len(image_files) >= 3:
                # Shuffle deterministic rồi chia
                shuffled = image_files.copy()
                rng.shuffle(shuffled)

                n = len(shuffled)
                train_end = int(n * self.split_ratio[0])
                val_end = train_end + int(n * self.split_ratio[1])

                # Đảm bảo mỗi split có ít nhất 1 ảnh
                train_end = max(1, train_end)
                val_end = max(train_end + 1, val_end)

                if self.split == "train":
                    selected = shuffled[:train_end]
                elif self.split == "val":
                    selected = shuffled[train_end:val_end]
                else:  # test
                    selected = shuffled[val_end:]
            else:
                # Không split hoặc quá ít ảnh → dùng tất cả
                selected = image_files

            for file_name in selected:
                img_path = os.path.join(identity_dir, file_name)
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError("Dataset rỗng hoặc không tìm thấy ảnh hợp lệ.")

        split_name = self.split or "all"
        print(f"[{split_name.upper()}] Loaded {len(self.samples)} images "
              f"from {self.num_classes} identities")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label