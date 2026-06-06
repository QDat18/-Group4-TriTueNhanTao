import os
from typing import Callable, Optional, Tuple, List, Dict, Union

from PIL import Image
from torch.utils.data import Dataset


class RMFRDDataset(Dataset):
    """
    Dataset Loader cho RMFRD/RWMFD.

    Hỗ trợ cấu trúc:
    root/
    ├── RWMFD_part_1/
    │   ├── 0000/
    │   ├── 0001/
    │   └── ...
    ├── RWMFD_part_2/
    └── RWMFD_part_3/

    Mỗi folder con là một danh tính.
    """

    def __init__(
        self,
        root_dir: Union[str, List[str]],
        transform: Optional[Callable] = None,
        max_classes: Optional[int] = None,
        max_images_per_class: Optional[int] = None
    ) -> None:
        if isinstance(root_dir, str):
            self.root_dirs = [root_dir]
        else:
            self.root_dirs = root_dir

        self.transform = transform
        self.max_classes = max_classes
        self.max_images_per_class = max_images_per_class

        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        self.valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

        self._load_samples()

    def _load_samples(self) -> None:
        identities = []

        for root in self.root_dirs:
            if not os.path.exists(root):
                print(f"[WARNING] Không tìm thấy thư mục: {root}")
                continue

            for folder in sorted(os.listdir(root)):
                folder_path = os.path.join(root, folder)

                if os.path.isdir(folder_path):
                    identity_key = f"{os.path.basename(root)}_{folder}"
                    identities.append((identity_key, folder_path))

        if self.max_classes is not None:
            identities = identities[:self.max_classes]

        for label, (identity_key, identity_dir) in enumerate(identities):
            self.class_to_idx[identity_key] = label

            image_files = [
                f for f in sorted(os.listdir(identity_dir))
                if f.lower().endswith(self.valid_exts)
            ]

            if self.max_images_per_class is not None:
                image_files = image_files[:self.max_images_per_class]

            for file_name in image_files:
                img_path = os.path.join(identity_dir, file_name)
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError("RMFRD dataset rỗng hoặc chưa giải nén đúng cấu trúc.")

        print(f"Loaded RMFRD images: {len(self.samples)}")
        print(f"Loaded RMFRD identities: {len(self.class_to_idx)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    from src.utils.transforms import get_train_transform

    dataset = RMFRDDataset(
        root_dir=[
            "dataset/RMFRDvaSMFRD/Real-World-Masked-Face-Dataset/RWMFD_part_1"
        ],
        transform=get_train_transform(),
        max_classes=10,
        max_images_per_class=5
    )

    image, label = dataset[0]

    print("=" * 60)
    print(f"Total images: {len(dataset)}")
    print(f"Total identities: {len(dataset.class_to_idx)}")
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    print("=" * 60)