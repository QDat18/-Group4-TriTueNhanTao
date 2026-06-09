import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    Lớp ArcFace Head (Additive Angular Margin Loss).

    Mục tiêu:
    - Thúc đẩy khoảng cách góc (angular margin) giữa các lớp (identities).
    - Làm embedding của cùng một người gần nhau hơn trên siêu cầu.
    - Làm embedding của những người khác nhau cách xa nhau hơn.
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5
    ) -> None:
        super().__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        # Trọng số đại diện cho các tâm lớp (class centroids)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # Ngưỡng góc để tránh tràn số góc khi cos(theta + m) vượt quá miền xác định
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 1. Tính toán cosine(theta) giữa embeddings và weights (các tâm lớp)
        # Cả hai đều được chuẩn hóa L2 nên dot product = cos(theta)
        cosine = F.linear(
            F.normalize(embeddings),
            F.normalize(self.weight)
        )

        # 2. Tính toán sine(theta) từ cosine(theta)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))

        # 3. Tính toán cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Bảo vệ khi góc theta lớn hơn pi - m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 4. Tạo mặt nạ One-hot đại diện cho nhãn đúng
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # 5. Áp dụng margin góc đối với nhãn đúng, giữ nguyên cosine đối với các nhãn sai
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 6. Nhân với tham số tỉ lệ (scale factor)
        logits *= self.scale

        return logits
