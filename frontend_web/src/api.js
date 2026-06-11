const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

async function request(endpoint, options = {}) {
  try {
    const resp = await fetch(`${API_BASE}${endpoint}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (err) {
    console.error(`API Error [${endpoint}]:`, err);
    return { error: err.message };
  }
}

// ── Dashboard ──
export const getDashboardStats = () => request('/dashboard/stats');
export const getAttendanceChart = (days = 30) => request(`/dashboard/attendance-chart?days=${days}`);
export const getDepartmentRanking = () => request('/dashboard/department-ranking');

// ── Employees ──
export const listEmployees = (search, department) => {
  const params = new URLSearchParams();
  if (search) params.set('search', search);
  if (department) params.set('department', department);
  return request(`/employees?${params}`);
};
export const getEmployee = (id) => request(`/employees/${id}`);
export const createEmployee = (data) => request('/employees', { method: 'POST', body: JSON.stringify(data) });
export const updateEmployee = (id, data) => request(`/employees/${id}`, { method: 'PUT', body: JSON.stringify(data) });
export const deleteEmployee = (id) => request(`/employees/${id}`, { method: 'DELETE' });

// ── Attendance ──
export const getAttendanceLogs = (params = {}) => {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v) q.set(k, v); });
  return request(`/attendance?${q}`);
};

// ── Embeddings ──
export const listEmbeddings = () => request('/embeddings');
export const deleteEmbedding = (id) => request(`/embeddings/${id}`, { method: 'DELETE' });
export const rebuildEmbeddings = () => request('/embeddings/rebuild', { method: 'POST' });
export const reloadEmbeddings = () => request('/embeddings/reload', { method: 'POST' });

// ── Devices ──
export const listDevices = () => request('/devices');
export const createDevice = (data) => request('/devices', { method: 'POST', body: JSON.stringify(data) });
export const toggleDevice = (id) => request(`/devices/${id}/toggle`, { method: 'PUT' });
export const deleteDevice = (id) => request(`/devices/${id}`, { method: 'DELETE' });

// ── Reports ──
export const getReportSummary = (period = 'month') => request(`/reports/summary?period=${period}`);
export const getReportByDepartment = () => request('/reports/by-department');

// ── Face Registration Process Control ──
export const startRegistration = (data) => request('/register/start', { method: 'POST', body: JSON.stringify(data) });
export const getRegistrationProgress = (employee_id) => request(`/register/progress?employee_id=${employee_id}`);
export const stopRegistration = () => request('/register/stop', { method: 'POST' });

// ── Settings ──
export const getSettings = () => request('/settings');
export const updateSettings = (data) => request('/settings', { method: 'PUT', body: JSON.stringify(data) });

// ── Model Info ──
export const getModelInfo = () => request('/model/info');
