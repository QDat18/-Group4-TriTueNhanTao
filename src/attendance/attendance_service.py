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
        self.current_local_date = self._local_now().date()
        self.load_embeddings()

    def _local_now(self):
        return datetime.now(timezone.utc) + timedelta(hours=7)

    def _local_day_bounds_utc(self, local_dt=None):
        local_dt = local_dt or self._local_now()
        start_local = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(days=1)
        return (
            (start_local - timedelta(hours=7)).isoformat(),
            (end_local - timedelta(hours=7)).isoformat()
        )

    def _reset_daily_state_if_needed(self):
        today = self._local_now().date()
        if today != self.current_local_date:
            self.last_check_in_cache.clear()
            self.current_local_date = today
            print(f"[AttendanceService] Reset daily attendance cache for {today}.")

    def _same_local_day(self, utc_naive_dt, local_dt=None):
        local_dt = local_dt or self._local_now()
        return (utc_naive_dt + timedelta(hours=7)).date() == local_dt.date()

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
        self._reset_daily_state_if_needed()
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        now_local = self._local_now()
        if employee_id in self.last_check_in_cache:
            last_time = self.last_check_in_cache[employee_id]
            if self._same_local_day(last_time, now_local):
                delta = now_utc - last_time
                remaining = config.COOLDOWN_SECONDS - delta.total_seconds()
                if remaining > 0:
                    return False, last_time, remaining
            else:
                self.last_check_in_cache.pop(employee_id, None)

        try:
            start_utc, end_utc = self._local_day_bounds_utc(now_local)
            response = (
                supabase
                .table("attendance_logs")
                .select("check_time")
                .eq("employee_id", employee_id)
                .gte("check_time", start_utc)
                .lt("check_time", end_utc)
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
            start_utc, end_utc = self._local_day_bounds_utc()
            response = (
                supabase
                .table("attendance_logs")
                .select("check_time, status")
                .eq("employee_id", employee_id)
                .gte("check_time", start_utc)
                .lt("check_time", end_utc)
                .order("check_time", desc=False)
                .execute()
            )
            return response.data or []
        except Exception:
            return []

    def save_attendance(self, employee_id, similarity, camera_id="CAM001"):
        from src.config import WORK_START_TIME, WORK_END_TIME, COOLDOWN_SECONDS

        self._reset_daily_state_if_needed()

        # 1. Get today's logs for the employee (UTC+7)
        today_logs = self._get_today_checkins(employee_id)
        
        # Check if they have checked in or checked out today
        has_checked_in = any(l.get("status") in ("SUCCESS", "LATE") for l in today_logs)
        has_checked_out = any(l.get("status") in ("CHECK_OUT", "EARLY_LEAVE") for l in today_logs)

        # Current local time (UTC+7)
        now_local = datetime.now(timezone.utc) + timedelta(hours=7)
        now_time = now_local.time()

        # Parse WORK_START_TIME and WORK_END_TIME
        try:
            start_h, start_m = map(int, WORK_START_TIME.split(":"))
            work_start_time = datetime.strptime(WORK_START_TIME, "%H:%M").time()
        except Exception:
            start_h, start_m = 8, 0
            work_start_time = datetime.strptime("08:00", "%H:%M").time()

        try:
            end_h, end_m = map(int, WORK_END_TIME.split(":"))
            work_end_time = datetime.strptime(WORK_END_TIME, "%H:%M").time()
        except Exception:
            end_h, end_m = 17, 30
            work_end_time = datetime.strptime("17:30", "%H:%M").time()

        work_start = now_local.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        work_end = now_local.replace(hour=end_h, minute=end_m, second=0, microsecond=0)

        midday_time = datetime.strptime("12:00", "%H:%M").time()

        # Determine status and metrics
        status = "SUCCESS"
        late_minutes = 0
        early_minutes = 0
        should_save = True

        if has_checked_in and has_checked_out:
            # Rule 7: Đã check-in và đã check-out
            status = "COMPLETED"
            should_save = False
        elif has_checked_in and not has_checked_out:
            # Đã check-in, chưa check-out
            if now_time < midday_time:
                # Rule 4: Từ chối check-out
                status = "REJECTED_CHECK_OUT"
                should_save = False
            else:
                if now_time < work_end_time:
                    # Rule 5: Check-out - Về sớm
                    status = "EARLY_LEAVE"
                    early_minutes = int((work_end - now_local).total_seconds() / 60)
                else:
                    # Rule 6: Check-out - Ra đúng giờ
                    status = "CHECK_OUT"
        else:
            # Chưa check-in
            if now_time < midday_time:
                if now_time <= work_start_time:
                    # Rule 1: Check-in - Đúng giờ
                    status = "SUCCESS"
                else:
                    # Rule 2: Check-in - Vào muộn
                    status = "LATE"
                    late_minutes = int((now_local - work_start).total_seconds() / 60)
            else:
                # Rule 3: Từ chối check-in
                status = "REJECTED_CHECK_IN"
                should_save = False

        # 2. Check Cooldown for DB-saving actions
        if should_save:
            # Check cooldown against last scan
            allowed, last_time, remaining = self.check_cooldown(employee_id)
            if not allowed:
                # Intelligent bypass: If last log was check-in (SUCCESS/LATE) and new status is check-out (CHECK_OUT/EARLY_LEAVE),
                # allow it as long as at least 15 seconds have passed (prevent camera double-triggering).
                is_transition = False
                if len(today_logs) > 0:
                    last_status = today_logs[-1].get("status")
                    last_was_in = last_status in ("SUCCESS", "LATE")
                    new_is_out = status in ("CHECK_OUT", "EARLY_LEAVE")
                    if last_was_in and new_is_out:
                        # Calculate time difference
                        last_log_time_str = today_logs[-1]["check_time"].replace("Z", "+00:00")
                        last_log_time = datetime.fromisoformat(last_log_time_str)
                        diff_seconds = (now_local - (last_log_time + timedelta(hours=7))).total_seconds()
                        if diff_seconds >= 15:
                            is_transition = True

                if not is_transition:
                    local_last = (last_time or datetime.now(timezone.utc).replace(tzinfo=None)) + timedelta(hours=7)
                    return {
                        "success": False,
                        "reason": "COOLDOWN",
                        "last_time": local_last.strftime("%H:%M:%S"),
                        "remaining": int(max(0, remaining))
                    }

        # 3. Save to Database if required
        if should_save:
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
