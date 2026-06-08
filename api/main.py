"""
FastAPI REST Server for Face Attendance Management System.
Shared backend for both Streamlit and Vite/React frontends.
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import sys
import time
from datetime import datetime, timedelta, timezone
import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.supabase_client import supabase
from api.auth import verify_api_key

app = FastAPI(
    title="Face Attendance API",
    description="REST API for Face Attendance Management System",
    version="1.0.0",
)

# Simple in-memory query cache to reduce load on Supabase during aggressive frontend polling
query_cache = {}

def get_cached_data(cache_key):
    now = time.time()
    if cache_key in query_cache:
        val, expiry = query_cache[cache_key]
        if now < expiry:
            return val
    return None

def set_cached_data(cache_key, val, ttl_seconds=3.0):
    query_cache[cache_key] = (val, time.time() + ttl_seconds)

# Ensure portrait directory exists and mount static files
os.makedirs("dataset/in-house", exist_ok=True)
app.mount("/api/portraits", StaticFiles(directory="dataset/in-house"), name="portraits")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════
# MODELS
# ════════════════════════════════════════

class EmployeeCreate(BaseModel):
    employee_id: str
    full_name: str
    department: str = "IT"
    position: str = "Employee"
    email: Optional[str] = None
    phone: Optional[str] = None


class EmployeeUpdate(BaseModel):
    full_name: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class DeviceCreate(BaseModel):
    device_id: str
    device_name: str
    location: str = ""
    is_active: bool = True


# ════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════

@app.get("/api/dashboard/stats")
def get_dashboard_stats():
    """Get overall dashboard statistics."""
    try:
        # Total employees
        emp_resp = supabase.table("employees").select("employee_id", count="exact").eq("is_active", True).execute()
        total_employees = emp_resp.count or len(emp_resp.data)

        # Today's attendance
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        att_resp = (
            supabase.table("attendance_logs")
            .select("employee_id, check_time, status")
            .gte("check_time", today_start)
            .execute()
        )

        today_logs = att_resp.data
        unique_today = set(log["employee_id"] for log in today_logs if log["status"] == "SUCCESS")
        present_today = len(unique_today)

        # Late arrivals (after 08:30)
        late_count = 0
        for log in today_logs:
            if log["status"] == "SUCCESS":
                check_time = datetime.fromisoformat(log["check_time"].replace("Z", "+00:00"))
                # Convert to local time (UTC+7)
                local_time = check_time + timedelta(hours=7)
                if local_time.hour > 8 or (local_time.hour == 8 and local_time.minute > 30):
                    late_count += 1

        absent_count = max(0, total_employees - present_today)

        return {
            "total_employees": total_employees,
            "present_today": present_today,
            "late_today": late_count,
            "absent_today": absent_count,
        }
    except Exception as e:
        return {
            "total_employees": 0,
            "present_today": 0,
            "late_today": 0,
            "absent_today": 0,
            "error": str(e),
        }


@app.get("/api/dashboard/attendance-chart")
def get_attendance_chart(days: int = 30):
    """Get attendance data for the last N days."""
    try:
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        resp = (
            supabase.table("attendance_logs")
            .select("employee_id, check_time, status")
            .gte("check_time", start_date)
            .order("check_time")
            .execute()
        )

        # Group by date
        daily = {}
        for log in resp.data:
            date = log["check_time"][:10]
            if date not in daily:
                daily[date] = set()
            if log["status"] == "SUCCESS":
                daily[date].add(log["employee_id"])

        chart_data = [
            {"date": date, "count": len(employees)}
            for date, employees in sorted(daily.items())
        ]

        return {"data": chart_data}
    except Exception as e:
        return {"data": [], "error": str(e)}


@app.get("/api/dashboard/department-ranking")
def get_department_ranking():
    """Get attendance ranking by department."""
    try:
        # Get all employees with department
        emp_resp = supabase.table("employees").select("employee_id, department").eq("is_active", True).execute()

        dept_map = {}
        for emp in emp_resp.data:
            dept = emp.get("department", "Unknown")
            if dept not in dept_map:
                dept_map[dept] = {"total": 0, "present": 0}
            dept_map[dept]["total"] += 1

        # Today's attendance
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        att_resp = (
            supabase.table("attendance_logs")
            .select("employee_id")
            .gte("check_time", today_start)
            .eq("status", "SUCCESS")
            .execute()
        )

        present_ids = set(log["employee_id"] for log in att_resp.data)

        # Map present employees to departments
        for emp in emp_resp.data:
            if emp["employee_id"] in present_ids:
                dept = emp.get("department", "Unknown")
                dept_map[dept]["present"] += 1

        ranking = [
            {
                "department": dept,
                "total": info["total"],
                "present": info["present"],
                "rate": round(info["present"] / max(1, info["total"]) * 100, 1),
            }
            for dept, info in dept_map.items()
        ]
        ranking.sort(key=lambda x: x["rate"], reverse=True)

        return {"data": ranking}
    except Exception as e:
        return {"data": [], "error": str(e)}


# ════════════════════════════════════════
# EMPLOYEES
# ════════════════════════════════════════

@app.get("/api/employees")
def list_employees(
    search: Optional[str] = None,
    department: Optional[str] = None,
):
    """List all employees with optional filters."""
    try:
        query = supabase.table("employees").select("*").eq("is_active", True)

        if department:
            query = query.eq("department", department)

        resp = query.order("employee_id").execute()

        data = resp.data
        if search:
            search_lower = search.lower()
            data = [
                e for e in data
                if search_lower in e["employee_id"].lower()
                or search_lower in (e.get("full_name", "") or "").lower()
            ]

        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/employees/{employee_id}")
def get_employee(employee_id: str):
    """Get employee details by ID."""
    try:
        resp = supabase.table("employees").select("*").eq("employee_id", employee_id).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="Employee not found")

        employee = resp.data[0]

        # Get face embedding info
        emb_resp = (
            supabase.table("face_embeddings")
            .select("image_count, updated_at")
            .eq("employee_id", employee_id)
            .execute()
        )

        if emb_resp.data:
            employee["image_count"] = emb_resp.data[0].get("image_count", 0)
            employee["embedding_updated_at"] = emb_resp.data[0].get("updated_at")
        else:
            employee["image_count"] = 0
            employee["embedding_updated_at"] = None

        return employee
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/employees", dependencies=[Depends(verify_api_key)])
def create_employee(employee: EmployeeCreate):
    """Create a new employee."""
    try:
        payload = employee.model_dump(exclude_none=True)
        supabase.table("employees").upsert(payload).execute()
        return {"message": "Employee created", "employee_id": employee.employee_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/employees/{employee_id}", dependencies=[Depends(verify_api_key)])
def update_employee(employee_id: str, update: EmployeeUpdate):
    """Update an employee."""
    try:
        payload = update.model_dump(exclude_none=True)
        if not payload:
            raise HTTPException(status_code=400, detail="No fields to update")

        supabase.table("employees").update(payload).eq("employee_id", employee_id).execute()
        return {"message": "Employee updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/employees/{employee_id}", dependencies=[Depends(verify_api_key)])
def delete_employee(employee_id: str):
    """Soft-delete an employee: set is_active=False, clean up biometric embeddings and local photos."""
    try:
        # 1. Soft-delete employee record (keeps details for historical attendance logs FK)
        supabase.table("employees").update({"is_active": False}).eq("employee_id", employee_id).execute()
        
        # 2. Hard-delete their biometric face embeddings (security and storage cleanup)
        supabase.table("face_embeddings").delete().eq("employee_id", employee_id).execute()
        
        # 3. Clean up physical dataset portraits folder
        portrait_dir = os.path.join("dataset", "in-house", employee_id)
        if os.path.exists(portrait_dir):
            import shutil
            try:
                shutil.rmtree(portrait_dir)
            except Exception as se:
                print(f"Warning: Could not remove local files for {employee_id}: {se}")

        # 4. Trigger memory reload in active realtime camera loop
        global realtime_system
        if realtime_system is not None:
            try:
                realtime_system.attendance_service.load_embeddings()
            except Exception as re:
                print(f"Warning: Could not reload embeddings in memory: {re}")

        return {"message": "Employee deactivated, biometric data and local portraits cleaned successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# ATTENDANCE LOGS
# ════════════════════════════════════════

@app.get("/api/attendance")
def get_attendance_logs(
    date: Optional[str] = None,
    department: Optional[str] = None,
    employee_id: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
):
    """Get attendance logs with filters."""
    cache_key = f"attendance_{date}_{department}_{employee_id}_{limit}"
    cached = get_cached_data(cache_key)
    if cached is not None:
        return cached

    try:
        query = supabase.table("attendance_logs").select("*")

        if employee_id:
            query = query.eq("employee_id", employee_id)

        if date:
            date_start = f"{date}T00:00:00+00:00"
            date_end = f"{date}T23:59:59+00:00"
            query = query.gte("check_time", date_start).lte("check_time", date_end)

        resp = query.order("check_time", desc=True).limit(limit).execute()

        logs = resp.data

        # If filtering by department, join with employees
        if department:
            emp_resp = supabase.table("employees").select("employee_id").eq("department", department).execute()
            dept_ids = set(e["employee_id"] for e in emp_resp.data)
            logs = [l for l in logs if l["employee_id"] in dept_ids]

        # Enrich with employee names (optimized query to fetch only required employees)
        emp_ids = list(set(l["employee_id"] for l in logs))
        if emp_ids:
            emp_resp = supabase.table("employees").select("employee_id, full_name, department").in_("employee_id", emp_ids).execute()
            emp_map = {e["employee_id"]: e for e in emp_resp.data}
            for log in logs:
                emp = emp_map.get(log["employee_id"], {})
                log["full_name"] = emp.get("full_name", "Unknown")
                log["department"] = emp.get("department", "Unknown")

        res = {"data": logs}
        set_cached_data(cache_key, res, ttl_seconds=3.0)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# EMBEDDINGS
# ════════════════════════════════════════

@app.get("/api/embeddings")
def list_embeddings():
    """List all face embeddings, deduplicated by employee_id (keeping the latest one)."""
    try:
        resp = supabase.table("face_embeddings").select(
            "employee_id, image_count, created_at, employees(full_name)"
        ).order("created_at", desc=True).execute()
        
        seen_ids = set()
        formatted = []
        for row in resp.data:
            emp_id = row["employee_id"]
            if emp_id in seen_ids:
                continue
            seen_ids.add(emp_id)
            
            emp = row.get("employees") or {}
            full_name = emp.get("full_name", "Unknown") if isinstance(emp, dict) else "Unknown"
            formatted.append({
                "employee_id": emp_id,
                "full_name": full_name,
                "image_count": row["image_count"],
                "updated_at": row.get("created_at")
            })
        return {"data": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/embeddings/{employee_id}", dependencies=[Depends(verify_api_key)])
def delete_embedding(employee_id: str):
    """Delete embedding for an employee."""
    try:
        supabase.table("face_embeddings").delete().eq("employee_id", employee_id).execute()
        return {"message": "Embedding deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# DEVICES / CAMERAS
# ════════════════════════════════════════

@app.get("/api/devices")
def list_devices():
    """List all devices/cameras."""
    try:
        resp = supabase.table("devices").select("*").execute()
        return {"data": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/devices", dependencies=[Depends(verify_api_key)])
def create_device(device: DeviceCreate):
    """Create or update a device."""
    try:
        payload = device.model_dump()
        supabase.table("devices").upsert(payload).execute()
        return {"message": "Device saved", "device_id": device.device_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/devices/{device_id}/toggle")
def toggle_device(device_id: str):
    """Toggle device active status."""
    try:
        resp = supabase.table("devices").select("is_active").eq("device_id", device_id).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="Device not found")
        new_status = not resp.data[0]["is_active"]
        supabase.table("devices").update({"is_active": new_status}).eq("device_id", device_id).execute()
        return {"message": "Toggled", "is_active": new_status}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/devices/{device_id}", dependencies=[Depends(verify_api_key)])
def delete_device(device_id: str):
    """Delete a device."""
    try:
        supabase.table("devices").delete().eq("device_id", device_id).execute()
        return {"message": "Device deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# REPORTS
# ════════════════════════════════════════

@app.get("/api/reports/summary")
def get_report_summary(
    period: str = Query(default="month", pattern="^(day|week|month)$"),
):
    """Get attendance report summary."""
    cache_key = f"summary_{period}"
    cached = get_cached_data(cache_key)
    if cached is not None:
        return cached

    try:
        now = datetime.now(timezone.utc)
        if period == "day":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Get total employees
        emp_resp = supabase.table("employees").select("employee_id", count="exact").eq("is_active", True).execute()
        total_employees = emp_resp.count or len(emp_resp.data)

        # Get attendance logs
        att_resp = (
            supabase.table("attendance_logs")
            .select("employee_id, check_time, status")
            .gte("check_time", start.isoformat())
            .execute()
        )

        logs = att_resp.data

        # Calculate days in period
        delta = now - start
        working_days = max(1, delta.days + 1)

        # Unique attendance days
        daily_present = {}
        late_set = set()
        for log in logs:
            if log["status"] == "SUCCESS":
                date = log["check_time"][:10]
                if date not in daily_present:
                    daily_present[date] = set()
                daily_present[date].add(log["employee_id"])

                # Check late
                check_time = datetime.fromisoformat(log["check_time"].replace("Z", "+00:00"))
                local_time = check_time + timedelta(hours=7)
                if local_time.hour > 8 or (local_time.hour == 8 and local_time.minute > 30):
                    late_set.add(f"{date}_{log['employee_id']}")

        total_possible = total_employees * working_days
        total_present = sum(len(v) for v in daily_present.values())
        total_late = len(late_set)
        total_absent = max(0, total_possible - total_present)

        res = {
            "period": period,
            "working_days": working_days,
            "total_employees": total_employees,
            "attendance_rate": round(total_present / max(1, total_possible) * 100, 1),
            "late_rate": round(total_late / max(1, total_present) * 100, 1),
            "absent_rate": round(total_absent / max(1, total_possible) * 100, 1),
            "daily_data": [
                {"date": date, "present": len(emps)}
                for date, emps in sorted(daily_present.items())
            ],
        }
        set_cached_data(cache_key, res, ttl_seconds=3.0)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/by-department")
def get_report_by_department():
    """Get attendance stats grouped by department."""
    try:
        emp_resp = supabase.table("employees").select("employee_id, department").eq("is_active", True).execute()

        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        att_resp = (
            supabase.table("attendance_logs")
            .select("employee_id")
            .gte("check_time", today_start)
            .eq("status", "SUCCESS")
            .execute()
        )

        present_ids = set(l["employee_id"] for l in att_resp.data)

        dept_stats = {}
        for emp in emp_resp.data:
            dept = emp.get("department", "Unknown")
            if dept not in dept_stats:
                dept_stats[dept] = {"total": 0, "present": 0}
            dept_stats[dept]["total"] += 1
            if emp["employee_id"] in present_ids:
                dept_stats[dept]["present"] += 1

        result = [
            {
                "department": dept,
                "total": info["total"],
                "present": info["present"],
                "absent": info["total"] - info["present"],
                "rate": round(info["present"] / max(1, info["total"]) * 100, 1),
            }
            for dept, info in sorted(dept_stats.items())
        ]

        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# FACE REGISTRATION PROCESS CONTROL
# ════════════════════════════════════════

import subprocess

active_process = None

class RegisterStartRequest(BaseModel):
    employee_id: str
    full_name: str
    department: str = "IT"
    position: str = "Employee"

@app.post("/api/register/start")
def start_registration(req: RegisterStartRequest):
    global active_process
    try:
        if active_process and active_process.poll() is None:
            active_process.terminate()
            active_process.wait()
            
        payload = {
            "employee_id": req.employee_id,
            "full_name": req.full_name,
            "department": req.department,
            "position": req.position
        }
        supabase.table("employees").upsert(payload).execute()
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        cmd = [
            sys.executable,
            "-m",
            "src.attendance.register_employee",
            "--employee_id", req.employee_id,
            "--full_name", req.full_name,
            "--department", req.department,
            "--position", req.position,
            "--max_images", "100"
        ]
        
        active_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=project_root,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/register/progress")
def get_registration_progress(employee_id: str):
    global active_process, realtime_system
    try:
        path = os.path.join("dataset", "in-house", employee_id)
        count = 0
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith(".jpg")])
            
        running = False
        if active_process:
            if active_process.poll() is None:
                running = True
            else:
                active_process = None
                # Auto reload embeddings in running live app
                if realtime_system is not None:
                    try:
                        realtime_system.attendance_service.load_embeddings()
                        print("Automatically reloaded embeddings after successful face registration.")
                    except Exception as re:
                        print(f"Error reloading embeddings: {re}")
            
        return {
            "count": count,
            "max_images": 100,
            "is_running": running
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register/stop")
def stop_registration():
    global active_process
    try:
        if active_process and active_process.poll() is None:
            active_process.terminate()
            active_process.wait()
            active_process = None
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embeddings/rebuild", dependencies=[Depends(verify_api_key)])
def rebuild_embeddings():
    """Trigger rebuild of embeddings and reload them into memory."""
    try:
        from src.attendance.build_embeddings import EmbeddingBuilder
        builder = EmbeddingBuilder()
        builder.run()
        
        # Reload embeddings in-memory
        global realtime_system
        if realtime_system is not None:
            realtime_system.attendance_service.load_embeddings()
            
        return {"status": "success", "message": "Rebuilt and reloaded embeddings successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embeddings/reload")
def reload_embeddings():
    """Reload embeddings from database to memory."""
    try:
        global realtime_system
        if realtime_system is not None:
            realtime_system.attendance_service.load_embeddings()
        return {"status": "success", "message": "Reloaded embeddings successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# SETTINGS CONFIGURATION
# ════════════════════════════════════════

class SettingsSchema(BaseModel):
    work_start_time: str
    allow_late_minutes: int
    cooldown_seconds: int
    recognition_threshold: float
    camera_source_type: str
    camera_webcam_index: int
    camera_ip_url: str

@app.get("/api/settings", response_model=SettingsSchema)
def get_settings():
    try:
        from src import config
        return {
            "work_start_time": config.WORK_START_TIME,
            "allow_late_minutes": config.ALLOW_LATE_MINUTES,
            "cooldown_seconds": config.COOLDOWN_SECONDS,
            "recognition_threshold": config.RECOGNITION_THRESHOLD,
            "camera_source_type": config.CAMERA_SOURCE_TYPE,
            "camera_webcam_index": config.CAMERA_WEBCAM_INDEX,
            "camera_ip_url": config.CAMERA_IP_URL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/settings", dependencies=[Depends(verify_api_key)])
def update_settings(req: SettingsSchema):
    try:
        config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(config_dir, "config.json")
        payload = {
            "work_start_time": req.work_start_time,
            "allow_late_minutes": req.allow_late_minutes,
            "cooldown_seconds": req.cooldown_seconds,
            "recognition_threshold": req.recognition_threshold,
            "camera_source_type": req.camera_source_type,
            "camera_webcam_index": req.camera_webcam_index,
            "camera_ip_url": req.camera_ip_url
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
            
        # Update in-memory variables to avoid server restart
        from src import config
        config.WORK_START_TIME = req.work_start_time
        config.ALLOW_LATE_MINUTES = req.allow_late_minutes
        config.COOLDOWN_SECONDS = req.cooldown_seconds
        config.RECOGNITION_THRESHOLD = req.recognition_threshold
        config.CAMERA_SOURCE_TYPE = req.camera_source_type
        config.CAMERA_WEBCAM_INDEX = req.camera_webcam_index
        config.CAMERA_IP_URL = req.camera_ip_url
        
        return {"message": "Settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/info")
def get_model_info():
    """Get active face recognition model details, metrics, and report."""
    try:
        from src import config
        import os
        
        model_path = config.FINAL_CHECKPOINT_PATH
        model_name = os.path.basename(model_path)
        
        # Set metrics based on model
        if "finetuned" in model_name or "rmfrd" in model_name:
            metrics = {
                "rank1": "7.11%",
                "rank5": "23.35%",
                "eer": "32.28%",
                "threshold": "0.44"
            }
        elif "warmup" in model_name:
            metrics = {
                "rank1": "4.57%",
                "rank5": "16.24%",
                "eer": "42.80%",
                "threshold": "0.11"
            }
        else:
            metrics = {
                "rank1": "—",
                "rank5": "—",
                "eer": "—",
                "threshold": "—"
            }
            
        # Try to read the evaluation report
        report_content = ""
        report_path = "evaluation_reports/afdb_masked_evaluation_report.md"
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
                
        return {
            "model_path": model_path,
            "model_name": model_name,
            "metrics": metrics,
            "report": report_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# CAMERA STREAM & REAL-TIME RECOGNITION
# ════════════════════════════════════════

from functools import lru_cache

@lru_cache(maxsize=1)
def get_realtime_system():
    """Lazy-init singleton for the realtime recognition system (thread-safe via lru_cache)."""
    global realtime_system
    from src.attendance.realtime_recognition import RealtimeRecognition
    realtime_system = RealtimeRecognition()
    return realtime_system

# Alias for backward-compat with references like `realtime_system.attendance_service`
realtime_system = None


@app.get("/api/attendance/current-face")
def get_current_face():
    """Return the person currently being recognized in the live camera feed."""
    try:
        system = get_realtime_system()
        # Snapshot active streams without holding the lock long
        with system.stream_lock:
            streams_snapshot = list(system.active_streams.values())

        best_match = None
        for stream_info in streams_snapshot:
            worker = stream_info.get("worker")
            if not worker:
                continue
            data = worker.get_draw_data()  # thread-safe via worker.lock
            for item in data:
                label = item.get("label", "")
                # Only show recognized employees (not Unknown/SPOOF/Error/empty)
                if not label or label in ("Unknown", "Error") or "SPOOF" in label:
                    continue
                # Parse label format: "Name - STATUS ..." or "Name - COOLDOWN (Xs)"
                parts = label.split(" - ", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    status_raw = parts[1].strip()
                    if "SUCCESS" in status_raw:
                        status = "SUCCESS"
                    elif "LATE" in status_raw:
                        status = "LATE"
                    elif "COOLDOWN" in status_raw:
                        status = "COOLDOWN"
                    else:
                        status = "UNKNOWN"
                    best_match = {
                        "full_name": name,
                        "status": status,
                        "label": label,
                        "liveness_score": round(float(item.get("liveness_score") or 0.0), 4)
                    }
                    break
            if best_match:
                break

        return {"data": best_match}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"data": None, "error": str(e)}


@app.get("/api/attendance/stream")
def get_attendance_stream(camera_id: Optional[str] = None):
    """Stream camera feed with real-time recognition."""
    try:
        from src import config
        system = get_realtime_system()
        
        # Determine camera source
        if camera_id is not None:
            camera_source = camera_id
        else:
            if config.CAMERA_SOURCE_TYPE == "ip_camera" and config.CAMERA_IP_URL:
                camera_source = config.CAMERA_IP_URL
            else:
                camera_source = config.CAMERA_WEBCAM_INDEX
        
        def frame_generator():
            for frame_bytes in system.run_gen(camera_id=camera_source):
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
                
        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════

@app.get("/api/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.on_event("startup")
def startup_event():
    """Pre-load RealtimeRecognition system on startup to avoid multiple redundant loads on concurrent initial requests."""
    print("\n[STARTUP] Pre-loading Realtime Recognition Model on server startup...")
    try:
        get_realtime_system()
        print("[STARTUP] Realtime Recognition Model loaded successfully.\n")
    except Exception as e:
        print(f"[STARTUP] ERROR: Failed to pre-load model on startup: {e}\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
