from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_vggface2 import VGGFace2Dataset


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

dataset = VGGFace2Dataset(
    root_dir="dataset/VGGFace2/train",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Number of classes:", len(dataset.class_to_idx))