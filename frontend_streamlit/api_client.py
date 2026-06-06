"""
HTTP client for communicating with FastAPI backend.
Used by all Streamlit pages.
"""

import requests

API_BASE = "http://localhost:8000/api"


def _get(endpoint, params=None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server. Run: uvicorn api.main:app --reload"}
    except Exception as e:
        return {"error": str(e)}


def _post(endpoint, json_data=None):
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=json_data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _put(endpoint, json_data=None):
    try:
        resp = requests.put(f"{API_BASE}{endpoint}", json=json_data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _delete(endpoint):
    try:
        resp = requests.delete(f"{API_BASE}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ── Dashboard ──
def get_dashboard_stats():
    return _get("/dashboard/stats")

def get_attendance_chart(days=30):
    return _get("/dashboard/attendance-chart", {"days": days})

def get_department_ranking():
    return _get("/dashboard/department-ranking")

# ── Employees ──
def list_employees(search=None, department=None):
    params = {}
    if search: params["search"] = search
    if department: params["department"] = department
    return _get("/employees", params)

def get_employee(employee_id):
    return _get(f"/employees/{employee_id}")

def create_employee(data):
    return _post("/employees", data)

def update_employee(employee_id, data):
    return _put(f"/employees/{employee_id}", data)

def delete_employee(employee_id):
    return _delete(f"/employees/{employee_id}")

# ── Attendance ──
def get_attendance_logs(date=None, department=None, employee_id=None, limit=100):
    params = {"limit": limit}
    if date: params["date"] = date
    if department: params["department"] = department
    if employee_id: params["employee_id"] = employee_id
    return _get("/attendance", params)

# ── Embeddings ──
def list_embeddings():
    return _get("/embeddings")

def delete_embedding(employee_id):
    return _delete(f"/embeddings/{employee_id}")

# ── Devices ──
def list_devices():
    return _get("/devices")

def create_device(data):
    return _post("/devices", data)

def toggle_device(device_id):
    return _put(f"/devices/{device_id}/toggle")

def delete_device(device_id):
    return _delete(f"/devices/{device_id}")

# ── Reports ──
def get_report_summary(period="month"):
    return _get("/reports/summary", {"period": period})

def get_report_by_department():
    return _get("/reports/by-department")
