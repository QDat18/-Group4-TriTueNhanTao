import torch

from src.models.face_recognition_model import (
    FaceRecognitionModel
)

model = FaceRecognitionModel()

dummy = torch.randn(
    1,
    3,
    112,
    112
)

embedding = model.get_embedding(
    dummy
)

print(
    embedding.shape
)