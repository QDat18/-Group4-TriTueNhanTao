import os
import cv2
import numpy as np
import onnxruntime as ort


class LivenessDetector:
    def __init__(
        self,
        model_path="checkpoints/anti_spoofing/MiniFASNetV2.onnx",
        threshold=0.60,
        input_size=(112, 112),
        enabled=True
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.input_size = input_size
        self.enabled = enabled

        self.session = None
        self.input_name = None

        if self.enabled and os.path.exists(self.model_path):
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]

            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )

            self.input_name = self.session.get_inputs()[0].name

            print(
                "[OK] MiniFASNet anti-spoofing loaded:",
                self.model_path
            )
            print(
                "[OK] Providers:",
                self.session.get_providers()
            )
        else:
            print(
                "[WARNING] MiniFASNet model not found. "
                "Fallback to heuristic liveness."
            )

    def preprocess(self, face_image):
        image = cv2.resize(
            face_image,
            self.input_size
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(np.float32)
        image = image / 255.0

        image = (image - 0.5) / 0.5

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        return image.astype(np.float32)

    def softmax(self, x):
        x = x - np.max(
            x,
            axis=1,
            keepdims=True
        )

        exp = np.exp(x)

        return exp / np.sum(
            exp,
            axis=1,
            keepdims=True
        )

    def check_liveness_by_model(self, face_image):
        inp = self.preprocess(face_image)

        outputs = self.session.run(
            None,
            {
                self.input_name: inp
            }
        )

        logits = outputs[0]

        probs = self.softmax(logits)

        probs = probs.flatten()

        if len(probs) == 2:
            spoof_score = float(probs[0])
            live_score = float(probs[1])
        else:
            live_score = float(np.max(probs))
            spoof_score = 1.0 - live_score

        is_live = live_score >= self.threshold

        return {
            "is_live": is_live,
            "score": live_score,
            "reason": (
                "Live Face"
                if is_live
                else "Spoof Face"
            ),
            "live_score": live_score,
            "spoof_score": spoof_score
        }

    def check_liveness_by_heuristic(self, face_image):
        gray = cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2GRAY
        )

        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        hsv = cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2HSV
        )

        v_channel = hsv[:, :, 2]

        _, max_val, _, _ = cv2.minMaxLoc(
            v_channel
        )

        hist_v = cv2.calcHist(
            [v_channel],
            [0],
            None,
            [256],
            [0, 256]
        )

        peak_ratio = hist_v.max() / hist_v.sum()

        is_live = True
        reason = "Authentic Face"
        score = 0.90

        if blur_score < 25.0:
            is_live = False
            reason = "Low quality / blur"
            score = 0.35

        elif peak_ratio > 0.18:
            is_live = False
            reason = "Strong reflection"
            score = 0.40

        elif max_val > 252 and blur_score < 45.0:
            is_live = False
            reason = "Overexposure"
            score = 0.45

        return {
            "is_live": is_live,
            "score": float(score),
            "reason": reason,
            "blur": float(blur_score),
            "peak_ratio": float(peak_ratio)
        }

    def check_liveness(self, face_image):
        if face_image is None or face_image.size == 0:
            return {
                "is_live": False,
                "score": 0.0,
                "reason": "No face image"
            }

        if not self.enabled:
            return {
                "is_live": True,
                "score": 1.0,
                "reason": "Liveness disabled"
            }

        if self.session is not None:
            try:
                return self.check_liveness_by_model(
                    face_image
                )
            except Exception as e:
                print(
                    "[WARNING] MiniFASNet failed, "
                    f"fallback heuristic: {e}"
                )

        return self.check_liveness_by_heuristic(
            face_image
        )