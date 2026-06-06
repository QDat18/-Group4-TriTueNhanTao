import { useEffect, useState } from 'react';
import { Download } from 'lucide-react';
import { getAttendanceLogs } from '../api';

export default function AttendanceLogs() {
  const [logs, setLogs] = useState([]);
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10));
  const [dept, setDept] = useState('');
  const [empId, setEmpId] = useState('');

  useEffect(() => {
    getAttendanceLogs({ date, department: dept || undefined, employee_id: empId || undefined, limit: 200 })
      .then(r => setLogs(r.data || []));
  }, [date, dept, empId]);

  const exportCSV = () => {
    const header = 'Thời gian,Mã NV,Họ tên,Phòng ban,Similarity,Status,Camera\n';
    const rows = logs.map(l =>
      `${l.check_time},${l.employee_id},${l.full_name || ''},${l.department || ''},${l.similarity || ''},${l.status},${l.camera_id || ''}`
    ).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `attendance_${date}.csv`;
    a.click();
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Attendance Logs</h1>
        <p>Nhật ký chấm công</p>
      </div>

      <div className="toolbar">
        <div className="input-group" style={{ marginBottom: 0 }}>
          <label>📅 Ngày</label>
          <input type="date" className="input" value={date} onChange={e => setDate(e.target.value)} style={{ width: 180 }} />
        </div>
        <div className="input-group" style={{ marginBottom: 0 }}>
          <label>🏢 Phòng ban</label>
          <select className="input" value={dept} onChange={e => setDept(e.target.value)} style={{ width: 160 }}>
            <option value="">Tất cả</option>
            {['IT', 'HR', 'Finance', 'Marketing', 'Sales'].map(d => <option key={d}>{d}</option>)}
          </select>
        </div>
        <div className="input-group" style={{ marginBottom: 0 }}>
          <label>👤 Mã NV</label>
          <input className="input" placeholder="NV001" value={empId} onChange={e => setEmpId(e.target.value)} style={{ width: 140 }} />
        </div>
        <div className="toolbar-spacer" />
        <button className="btn btn-secondary" onClick={exportCSV} style={{ alignSelf: 'flex-end' }}>
          <Download size={14} /> Xuất CSV
        </button>
      </div>

      <div className="card">
        <table className="data-table">
          <thead>
            <tr><th>⏰ Thời gian</th><th>🆔 Mã NV</th><th>👤 Họ tên</th><th>🏢 Phòng ban</th><th>📊 Similarity</th><th>✅ Status</th><th>🎥 Camera</th></tr>
          </thead>
          <tbody>
            {logs.map((log, i) => (
              <tr key={i}>
                <td style={{ fontFamily: 'monospace', fontSize: '0.82rem' }}>{(log.check_time || '').replace('T', ' ').slice(0, 19)}</td>
                <td><strong style={{ color: 'var(--accent-info)' }}>{log.employee_id}</strong></td>
                <td>{log.full_name || '—'}</td>
                <td><span className="badge badge-info">{log.department || '—'}</span></td>
                <td style={{ fontFamily: 'monospace' }}>{log.similarity?.toFixed(2) || '—'}</td>
                <td><span className={`badge ${log.status === 'SUCCESS' ? 'badge-success' : 'badge-danger'}`}>{log.status}</span></td>
                <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{log.camera_id || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {logs.length === 0 && <div className="empty-state"><div className="empty-state-icon">📋</div><p>Không có bản ghi chấm công</p></div>}
      </div>
      <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>Tổng: {logs.length} bản ghi</p>
    </div>
  );
}
