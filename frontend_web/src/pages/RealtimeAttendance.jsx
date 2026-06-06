import { useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { getAttendanceLogs } from '../api';

export default function RealtimeAttendance() {
  const [logs, setLogs] = useState([]);

  const load = () => getAttendanceLogs({ limit: 20 }).then(r => setLogs(r.data || []));
  useEffect(() => { load(); }, []);

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Realtime Attendance</h1>
        <p>Chấm công thời gian thực</p>
      </div>

      <div className="grid-2">
        {/* Camera Feed */}
        <div className="card">
          <div className="card-header"><span className="card-title">🎥 Camera Feed</span></div>
          <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-md)', height: 350, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem', border: '2px dashed var(--border-glow)' }}>
            <div style={{ fontSize: '3rem' }}>📹</div>
            <p style={{ color: 'var(--text-muted)' }}>Camera Preview</p>
          </div>
        </div>

        {/* Recognition Result */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
          <div className="card-header" style={{ width: '100%' }}><span className="card-title">🎯 Kết Quả Nhận Diện</span></div>
          <div style={{ fontSize: '4rem', margin: '1rem 0' }}>👤</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-info)' }}>—</div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: '0.5rem 0' }}>Đang chờ nhận diện...</div>
          <div style={{ display: 'flex', gap: '2rem', marginTop: '1rem' }}>
            <div><div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>SIMILARITY</div><div style={{ fontWeight: 600 }}>—</div></div>
            <div><div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>STATUS</div><div style={{ fontWeight: 600 }}>—</div></div>
            <div><div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>TIME</div><div style={{ fontWeight: 600 }}>—</div></div>
          </div>
        </div>
      </div>

      {/* Live Log */}
      <div className="card" style={{ marginTop: '1.25rem' }}>
        <div className="card-header">
          <span className="card-title">📋 Log Thời Gian Thực</span>
          <button className="btn btn-sm btn-secondary" onClick={load}><RefreshCw size={14} /> Refresh</button>
        </div>
        {logs.length > 0 ? (
          <table className="data-table">
            <thead><tr><th>Thời gian</th><th>Mã NV</th><th>Họ tên</th><th>Similarity</th><th>Status</th></tr></thead>
            <tbody>
              {logs.slice(0, 10).map((log, i) => (
                <tr key={i}>
                  <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>{(log.check_time || '').slice(11, 19)}</td>
                  <td><strong style={{ color: 'var(--accent-info)' }}>{log.employee_id}</strong></td>
                  <td>{log.full_name || '—'}</td>
                  <td style={{ fontFamily: 'monospace' }}>{log.similarity?.toFixed(2) || '—'}</td>
                  <td><span className={`badge ${log.status === 'SUCCESS' ? 'badge-success' : 'badge-danger'}`}>{log.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="empty-state"><p>Chưa có log chấm công</p></div>
        )}
      </div>
    </div>
  );
}
