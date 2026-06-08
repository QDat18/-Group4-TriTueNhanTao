from torchvision import transforms


def get_train_transform():
    """
    Transform dùng cho tập train.
    Có augmentation nhẹ để tăng độ đa dạng dữ liệu.
    """
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2
        ),
        transforms.GaussianBlur(
            kernel_size=3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.15)
        )
    ])


def get_val_transform():
    """
    Transform dùng cho validation/test.
    Không augmentation để đánh giá ổn định.
    """
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])