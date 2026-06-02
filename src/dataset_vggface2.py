import os
from typing import Callable, Optional, Tuple, List, Dict

from PIL import Image
from torch.utils.data import Dataset


class VGGFace2Dataset(Dataset):
    """
    Dataset loader cho VGGFace2.

    Mỗi thư mục con tương ứng với một danh tính.
    """
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        max_classes: Optional[int] = None,
        max_images_per_class: Optional[int] = None
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.max_classes = max_classes
        self.max_images_per_class = max_images_per_class

        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}

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

        for label, identity in enumerate(identities):
            self.class_to_idx[identity] = label
            identity_dir = os.path.join(self.root_dir, identity)

            image_files = [
                f for f in os.listdir(identity_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if self.max_images_per_class is not None:
                image_files = image_files[:self.max_images_per_class]

            for file_name in image_files:
                img_path = os.path.join(identity_dir, file_name)
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError("Dataset rỗng hoặc không tìm thấy ảnh hợp lệ.")

        print(f"Loaded {len(self.samples)} images")
        print(f"Loaded {len(self.class_to_idx)} identities")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label