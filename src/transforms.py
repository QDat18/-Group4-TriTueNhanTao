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
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
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