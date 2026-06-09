import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src import config


class FaceEmbeddingNet(nn.Module):
    """
    Backbone trích xuất đặc trưng khuôn mặt sử dụng ResNet50 làm nền tảng,
    chiếu lên vector embedding 512 chiều, chuẩn hóa BN và L2.
    """

    def __init__(self, embedding_size: int = 512, pretrained: bool = False) -> None:
        super().__init__()

        # Sử dụng weights của ImageNet nếu pretrained=True
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        backbone = models.resnet50(weights=weights)

        # Loại bỏ lớp Fully Connected (FC) cuối cùng của ResNet50
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Ánh xạ từ đặc trưng 2048 chiều của ResNet50 về 512 chiều (embedding)
        self.embedding_layer = nn.Linear(2048, embedding_size)

        # Batch Normalization giúp embedding ổn định hơn
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trích xuất đặc trưng
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)

        # Dự chiếu và chuẩn hóa BN
        embeddings = self.embedding_layer(features)
        embeddings = self.bn(embeddings)

        # L2 Normalization để tính toán Cosine Similarity trực tiếp bằng dot product
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class FaceRecognitionModel:
    """
    Wrapper tải checkpoint mô hình và thực hiện trích xuất embedding cho hệ thống nhận diện.
    """

    def __init__(self, checkpoint_path: str = None, device: str = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceEmbeddingNet(embedding_size=512, pretrained=False)

        # Xác định checkpoint để load
        if checkpoint_path is None:
            # Các đường dẫn checkpoint có thể có trong hệ thống, sắp xếp theo thứ tự ưu tiên
            possible_paths = [
                # 1. Đường dẫn config.FINAL_CHECKPOINT_PATH (arcface_vggface2_warmup_lite.pth)
                getattr(config, "FINAL_CHECKPOINT_PATH", None),
                # 2. Checkpoint lite mặc định
                os.path.join(config.CHECKPOINT_DIR, "arcface_vggface2_warmup_lite.pth"),
                # 3. Checkpoint full mặc định
                os.path.join(config.CHECKPOINT_DIR, "arcface_vggface2_warmup.pth"),
                # 4. Checkpoint fine-tuned RMFRD lite
                os.path.join(config.CHECKPOINT_DIR, "arcface_rmfrd_finetuned_lite.pth"),
                # 5. Checkpoint fine-tuned RMFRD full
                os.path.join(config.CHECKPOINT_DIR, "arcface_rmfrd_finetuned.pth"),
            ]
            for path in possible_paths:
                if path and os.path.exists(path):
                    checkpoint_path = path
                    break

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading FaceRecognitionModel checkpoint from: {checkpoint_path}")
            # Đọc file checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Phân tách cấu trúc checkpoint
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load vào mô hình (chỉ nạp các trọng số của backbone)
            # Dùng strict=False để bỏ qua trọng số của ArcFace Head (nếu có trong checkpoint)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"[WARNING] Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                # Trọng số unexpected thường là của arcface_head, điều này là bình thường
                non_head_unexpected = [k for k in unexpected_keys if "weight" not in k]
                if non_head_unexpected:
                    print(f"[WARNING] Unexpected keys when loading checkpoint: {non_head_unexpected}")
        else:
            print("[WARNING] No face recognition checkpoint found. Model weights are uninitialized!")

        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất vector đặc trưng (embedding) từ ảnh khuôn mặt đầu vào.
        
        Args:
            image: Tensor ảnh [B, 3, 112, 112] hoặc [3, 112, 112].
            
        Returns:
            Tensor embedding 512 chiều nằm trên CPU.
        """
        # Đảm bảo có chiều Batch
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            embedding = self.model(image)

        return embedding.cpu()
