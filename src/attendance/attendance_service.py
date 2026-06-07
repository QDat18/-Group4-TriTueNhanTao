import numpy as np
import faiss
from datetime import datetime, timedelta, timezone

from src.database.supabase_client import supabase
from src import config


class AttendanceService:

    def __init__(self):

        self.employee_embeddings = {}
        self.employee_ids_list = []  # Ordered list for FAISS index mapping
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

            self.employee_embeddings[
                row["employee_id"]
            ] = {
                "full_name": full_name,
                "embedding": np.array(
                    row["embedding_vector"],
                    dtype=np.float32
                )
            }

        print(
            f"Loaded embeddings: "
            f"{len(self.employee_embeddings)}"
        )

        self._build_faiss_index()

    def _build_faiss_index(self):
        """Build FAISS index for fast nearest neighbor search."""
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
        self.faiss_index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine after L2 norm
        self.faiss_index.add(vectors)

        print(f"FAISS index built: {self.faiss_index.ntotal} vectors, dim={dim}")

    def cosine_similarity(
        self,
        emb1,
        emb2
    ):

        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)

        return float(
            np.dot(
                emb1,
                emb2
            )
        )

    def find_best_match(
        self,
        embedding
    ):
        """Find best matching employee using FAISS index (O(1) vs O(n))."""
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

    def check_cooldown(
        self,
        employee_id
    ):
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        if employee_id in self.last_check_in_cache:
            last_time = self.last_check_in_cache[employee_id]
            delta = now_utc - last_time
            remaining = config.COOLDOWN_SECONDS - delta.total_seconds()
            if remaining > 0:
                return False, last_time, remaining

        # Query Supabase once to populate/verify the cache if not in memory or expired
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
            # Fall back to assuming allowed to avoid blocking user check-in
            return True, None, 0

    def save_attendance(
        self,
        employee_id,
        similarity,
        camera_id="CAM001"
    ):
        allowed, last_time, remaining = self.check_cooldown(employee_id)
        if not allowed:
            # Convert last check time to local timezone (GMT+7) for human readability
            local_last = (last_time or datetime.now(timezone.utc).replace(tzinfo=None)) + timedelta(hours=7)
            return {
                "success": False,
                "reason": "COOLDOWN",
                "last_time": local_last.strftime("%H:%M:%S"),
                "remaining": int(max(0, remaining))
            }

        # Calculate late or on-time status based on config
        from src.config import WORK_START_TIME, ALLOW_LATE_MINUTES
        now = datetime.now()
        
        # Parse start hour and minute
        start_h, start_m = map(int, WORK_START_TIME.split(":"))
        limit_time = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0) + timedelta(minutes=ALLOW_LATE_MINUTES)
        
        status = "SUCCESS"
        late_minutes = 0
        
        # Only mark as LATE if checking in during the morning after the allowance limit
        if now > limit_time and now.hour < 12:
            status = "LATE"
            late_minutes = int((now - limit_time).total_seconds() / 60)

        payload = {
            "employee_id":
                employee_id,
            "similarity":
                similarity,
            "camera_id":
                camera_id,
            "status":
                status
        }

        try:
            (
                supabase
                .table("attendance_logs")
                .insert(payload)
                .execute()
            )
            # Update cache on successful database save
            self.last_check_in_cache[employee_id] = datetime.now(timezone.utc).replace(tzinfo=None)
        except Exception as e:
            print(f"Error saving attendance to DB: {e}")

        return {
            "success": True,
            "status": status,
            "late_minutes": late_minutes,
            "check_time": now.strftime("%H:%M:%S")
        }