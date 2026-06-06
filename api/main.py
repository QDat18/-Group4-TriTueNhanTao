"""
FastAPI REST Server for Face Attendance Management System.
Shared backend for both Streamlit and Vite/React frontends.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.supabase_client import supabase

app = FastAPI(
    title="Face Attendance API",
    description="REST API for Face Attendance Management System",
    version="1.0.0",
)

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


@app.post("/api/employees")
def create_employee(employee: EmployeeCreate):
    """Create a new employee."""
    try:
        payload = employee.model_dump(exclude_none=True)
        supabase.table("employees").upsert(payload).execute()
        return {"message": "Employee created", "employee_id": employee.employee_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/employees/{employee_id}")
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


@app.delete("/api/employees/{employee_id}")
def delete_employee(employee_id: str):
    """Soft-delete an employee (set is_active=False)."""
    try:
        supabase.table("employees").update({"is_active": False}).eq("employee_id", employee_id).execute()
        return {"message": "Employee deactivated"}
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

        # Enrich with employee names
        emp_ids = list(set(l["employee_id"] for l in logs))
        if emp_ids:
            emp_resp = supabase.table("employees").select("employee_id, full_name, department").execute()
            emp_map = {e["employee_id"]: e for e in emp_resp.data}
            for log in logs:
                emp = emp_map.get(log["employee_id"], {})
                log["full_name"] = emp.get("full_name", "Unknown")
                log["department"] = emp.get("department", "Unknown")

        return {"data": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# EMBEDDINGS
# ════════════════════════════════════════

@app.get("/api/embeddings")
def list_embeddings():
    """List all face embeddings."""
    try:
        resp = supabase.table("face_embeddings").select(
            "employee_id, full_name, image_count, updated_at"
        ).execute()
        return {"data": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/embeddings/{employee_id}")
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


@app.post("/api/devices")
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


@app.delete("/api/devices/{device_id}")
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
    period: str = Query(default="month", regex="^(day|week|month)$"),
):
    """Get attendance report summary."""
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

        return {
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
# HEALTH CHECK
# ════════════════════════════════════════

@app.get("/api/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
