import numpy as np
import faiss
from datetime import datetime, timedelta, timezone

from src.database.supabase_client import supabase
from src import config


class AttendanceService:

    def __init__(self):
        self.employee_embeddings = {}
        self.employee_ids_list = []
        self.faiss_index = None
        self.last_check_in_cache = {}
        self.load_embeddings()

    def load_embeddings(self):
        response = (
            supabase
            .table("face_embeddings")
            .select("employee_id, embedding_vector, employees(full_name, is_active)")
            .execute()
        )

        self.employee_embeddings = {}

        for row in response.data:
            emp = row.get("employees") or {}
            is_active = emp.get("is_active", True) if isinstance(emp, dict) else True
            if not is_active:
                continue

            full_name = emp.get("full_name", "Unknown") if isinstance(emp, dict) else "Unknown"

            self.employee_embeddings[row["employee_id"]] = {
                "full_name": full_name,
                "embedding": np.array(row["embedding_vector"], dtype=np.float32)
            }

        print(f"Loaded embeddings: {len(self.employee_embeddings)}")
        self._build_faiss_index()

    def _build_faiss_index(self):
        if len(self.employee_embeddings) == 0:
            self.faiss_index = None
            self.employee_ids_list = []
            return

        self.employee_ids_list = list(self.employee_embeddings.keys())
        vectors = np.array(
            [self.employee_embeddings[eid]["embedding"] for eid in self.employee_ids_list],
            dtype=np.float32
        )
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(vectors)

        print(f"FAISS index built: {self.faiss_index.ntotal} vectors, dim={dim}")

    def cosine_similarity(self, emb1, emb2):
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return float(np.dot(emb1, emb2))

    def find_best_match(self, embedding):
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return None

        query = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        scores, indices = self.faiss_index.search(query, 1)
        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])

        if best_score < config.RECOGNITION_THRESHOLD:
            return None

        best_employee = self.employee_ids_list[best_idx]
        return {
            "employee_id": best_employee,
            "full_name": self.employee_embeddings[best_employee]["full_name"],
            "similarity": best_score
        }

    def check_cooldown(self, employee_id):
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        if employee_id in self.last_check_in_cache:
            last_time = self.last_check_in_cache[employee_id]
            delta = now_utc - last_time
            remaining = config.COOLDOWN_SECONDS - delta.total_seconds()
            if remaining > 0:
                return False, last_time, remaining

        try:
            response = (
                supabase
                .table("attendance_logs")
                .select("check_time")
                .eq("employee_id", employee_id)
                .order("check_time", desc=True)
                .limit(1)
                .execute()
            )

            if len(response.data) == 0:
                self.last_check_in_cache[employee_id] = datetime.min
                return True, None, 0

            check_time_str = response.data[0]["check_time"].replace("Z", "+00:00")
            last_time = datetime.fromisoformat(check_time_str).replace(tzinfo=None)
            delta = now_utc - last_time
            remaining = config.COOLDOWN_SECONDS - delta.total_seconds()

            self.last_check_in_cache[employee_id] = last_time
            return delta.total_seconds() > config.COOLDOWN_SECONDS, last_time, remaining
        except Exception as e:
            print(f"Error checking cooldown in DB: {e}")
            return True, None, 0

    def _get_today_checkins(self, employee_id):
        """Lấy số lần chấm công hôm nay của nhân viên."""
        try:
            now_utc = datetime.now(timezone.utc)
            today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=7)
            response = (
                supabase
                .table("attendance_logs")
                .select("check_time, status")
                .eq("employee_id", employee_id)
                .gte("check_time", today_start.isoformat())
                .order("check_time", desc=False)
                .execute()
            )
            return response.data or []
        except Exception:
            return []

    def save_attendance(self, employee_id, similarity, camera_id="CAM001"):
        allowed, last_time, remaining = self.check_cooldown(employee_id)
        if not allowed:
            local_last = (last_time or datetime.now(timezone.utc).replace(tzinfo=None)) + timedelta(hours=7)
            return {
                "success": False,
                "reason": "COOLDOWN",
                "last_time": local_last.strftime("%H:%M:%S"),
                "remaining": int(max(0, remaining))
            }

        from src.config import WORK_START_TIME, WORK_END_TIME, ALLOW_LATE_MINUTES, ALLOW_EARLY_MINUTES

        # Thời gian hiện tại theo UTC+7
        now_local = datetime.now(timezone.utc) + timedelta(hours=7)

        # Parse giờ vào / ra
        start_h, start_m = map(int, WORK_START_TIME.split(":"))
        try:
            end_h, end_m = map(int, WORK_END_TIME.split(":"))
        except Exception:
            end_h, end_m = 17, 30

        work_start  = now_local.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        work_end    = now_local.replace(hour=end_h,   minute=end_m,   second=0, microsecond=0)
        late_limit  = work_start + timedelta(minutes=ALLOW_LATE_MINUTES)
        early_limit = work_end   - timedelta(minutes=ALLOW_EARLY_MINUTES)

        # Đếm số lần đã chấm công hôm nay
        today_logs = self._get_today_checkins(employee_id)
        checkin_count = len(today_logs)

        status = "SUCCESS"
        late_minutes = 0
        early_minutes = 0

        if checkin_count == 0:
            # ── Lần đầu: VÀO LÀM ──────────────────────────────
            if now_local > late_limit:
                status = "LATE"
                late_minutes = int((now_local - late_limit).total_seconds() / 60)
            else:
                status = "SUCCESS"
        else:
            # ── Lần 2 trở đi: RA VỀ ───────────────────────────
            if now_local < early_limit:
                status = "EARLY_LEAVE"
                early_minutes = int((early_limit - now_local).total_seconds() / 60)
            else:
                status = "CHECK_OUT"

        payload = {
            "employee_id": employee_id,
            "similarity":  similarity,
            "camera_id":   camera_id,
            "status":      status
        }

        try:
            supabase.table("attendance_logs").insert(payload).execute()
            self.last_check_in_cache[employee_id] = datetime.now(timezone.utc).replace(tzinfo=None)
        except Exception as e:
            print(f"Error saving attendance to DB: {e}")

        return {
            "success":       True,
            "status":        status,
            "late_minutes":  late_minutes,
            "early_minutes": early_minutes,
            "check_time":    now_local.strftime("%H:%M:%S")
        }