from insightface.app import FaceAnalysis
from src.config import DET_SIZE
import numpy as np
import cv2

class InsightFaceModel:
    def __init__(self, model_name='buffalo_l', ctx_id=0):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)

    def get(self, img):
        return self.app.get(img)

    def get_embedding(self, img):
        try:
            faces = self.app.get(img)
            if not faces:
                return None
            return faces[0].normed_embedding
        except Exception:
            return None

    def get_embedding_from_aligned(self, img):
        try:
            if img is None or img.size == 0:
                return None
            faces = self.app.get(img)
            if faces:
                return faces[0].normed_embedding
            resized = cv2.resize(img, (112, 112))
            faces = self.app.get(resized)
            if faces:
                return faces[0].normed_embedding
            return None
        except Exception:
            return None

    def get_embedding_from_face(self, face_img):
        try:
            if face_img is None:
                return None
            if hasattr(face_img, "normed_embedding"):
                return np.asarray(face_img.normed_embedding, dtype=np.float32)
            return self.get_embedding_from_aligned(face_img)
        except Exception:
            return None
