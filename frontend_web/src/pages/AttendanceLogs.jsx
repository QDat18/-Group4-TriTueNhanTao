import { useEffect, useState } from 'react';
import { Download, RefreshCw, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react';
import { getAttendanceLogs } from '../api';

function formatTime(value) {
  if (!value) return '—';
  return String(value).replace('T', ' ').slice(0, 19);
}

function formatSimilarity(value) {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (Number.isNaN(num)) return value;
  return num <= 1 ? `${(num * 100).toFixed(1)}%` : num.toFixed(2);
}

function normalizeStatus(status) {
  const raw = String(status || '').toUpperCase();

  if (['SUCCESS', 'OK', 'CHECKED_IN', 'PRESENT'].includes(raw)) {
    return {
      label: 'Thành công',
      className: 'badge-success',
      icon: <CheckCircle2 size={13} />
    };
  }

  if (['SPOOF', 'FAKE', 'LIVENESS_FAIL'].includes(raw)) {
    return {
      label: 'Từ chối - nghi giả mạo',
      className: 'badge-danger',
      icon: <XCircle size={13} />
    };
  }

  if (['UNKNOWN', 'NOT_FOUND', 'LOW_SIMILARITY'].includes(raw)) {
    return {
      label: 'Không nhận diện được',
      className: 'badge-warning',
      icon: <AlertTriangle size={13} />
    };
  }

  return {
    label: status || 'Chưa rõ',
    className: 'badge-info',
    icon: <AlertTriangle size={13} />
  };
}

function buildReason(log) {
  if (log.message) return log.message;
  if (log.reason) return log.reason;
  if (log.error) return log.error;

  const status = String(log.status || '').toUpperCase();

  if (status === 'SUCCESS') {
    return 'Đã ghi nhận chấm công thành công.';
  }

  if (['UNKNOWN', 'NOT_FOUND', 'LOW_SIMILARITY'].includes(status)) {
    return 'Không đủ độ tương đồng hoặc chưa có vector khuôn mặt.';
  }

  if (['SPOOF', 'FAKE', 'LIVENESS_FAIL'].includes(status)) {
    return 'Không đạt kiểm tra chống giả mạo/liveness.';
  }

  return 'Chưa có mô tả chi tiết từ backend.';
}

export default function AttendanceLogs() {
  const [logs, setLogs] = useState([]);
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10));
  const [dept, setDept] = useState('');
  const [empId, setEmpId] = useState('');
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState(null);

  const loadLogs = () => {
    setLoading(true);
    setNotice(null);

    getAttendanceLogs({
      date,
      department: dept || undefined,
      employee_id: empId.trim() || undefined,
      limit: 200
    })
      .then(res => {
        if (res?.error) {
          throw new Error(res.error);
        }

        const rows = res.data || [];
        setLogs(rows);
        setNotice({
          type: 'success',
          message: `Đã tải ${rows.length} bản ghi chấm công.`
        });
      })
      .catch(err => {
        setLogs([]);
        setNotice({
          type: 'error',
          message: `Không tải được nhật ký: ${err.message || 'Lỗi không xác định'}`
        });
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    loadLogs();
  }, [date, dept, empId]);

  const exportCSV = () => {
    const header = 'Thời gian,Mã NV,Họ tên,Phòng ban,Similarity,Status,Lý do,Camera\n';
    const rows = logs.map(l => [
      formatTime(l.check_time),
      l.employee_id || '',
      l.full_name || '',
      l.department || '',
      formatSimilarity(l.similarity),
      normalizeStatus(l.status).label,
      buildReason(l),
      l.camera_id || ''
    ].map(v => `"${String(v).replaceAll('"', '""')}"`).join(',')).join('\n');

    const blob = new Blob(['\ufeff' + header + rows], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `attendance_${date}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);

    setNotice({
      type: 'success',
      message: `Đã xuất file CSV attendance_${date}.csv.`
    });
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Nhật ký chấm công</h1>
        <p>Theo dõi lịch sử, trạng thái và lý do ghi nhận chấm công</p>
      </div>

      {notice && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.55rem',
          marginBottom: '1rem',
          padding: '0.75rem 0.95rem',
          borderRadius: 12,
          border: `1px solid ${notice.type === 'success' ? '#bbf7d0' : '#fecaca'}`,
          background: notice.type === 'success' ? '#f0fdf4' : '#fef2f2',
          color: notice.type === 'success' ? '#15803d' : '#b91c1c',
          fontSize: '0.82rem',
          fontWeight: 650
        }}>
          {notice.type === 'success' ? '✅' : '❌'} {notice.message}
        </div>
      )}

      <div className="toolbar">
        <div className="input-group" style={{ marginBottom: 0 }}>
          <label>📅 Ngày</label>
          <input type="date" className="input" value={date} onChange={e => setDate(e.target.value)} style={{ width: 180 }} />
        </div>
        <div className="input-group" style={{ marginBottom: 0 }}>
          <label>🏢 Phòng ban</label>
          <select className="input" value={dept} onChange={e => setDept(e.target.value)} style={{ width: 160 }}>
            <option value="">Tất cả</option>
            {['IT', 'HR', 'Finance', 'Marketing', 'Sales', 'Admin'].map(d => <option key={d}>{d}</option>)}
          </select>
        </div>
        <div className="input-group" style={{ marginBottom: 0 }}>
          <label>👤 Mã NV</label>
          <input className="input" placeholder="NV001" value={empId} onChange={e => setEmpId(e.target.value)} style={{ width: 140 }} />
        </div>
        <div className="toolbar-spacer" />
        <button className="btn btn-secondary" onClick={loadLogs} disabled={loading} style={{ alignSelf: 'flex-end' }}>
          <RefreshCw size={14} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} /> Làm mới
        </button>
        <button className="btn btn-secondary" onClick={exportCSV} disabled={logs.length === 0} style={{ alignSelf: 'flex-end' }}>
          <Download size={14} /> Xuất CSV
        </button>
      </div>

      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>⏰ Thời gian</th>
              <th>🆔 Mã NV</th>
              <th>👤 Họ tên</th>
              <th>🏢 Phòng ban</th>
              <th>📊 Similarity</th>
              <th>✅ Trạng thái</th>
              <th>📝 Ghi chú</th>
              <th>🎥 Camera</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log, i) => {
              const statusView = normalizeStatus(log.status);

              return (
                <tr key={log.id || `${log.employee_id}-${log.check_time}-${i}`}>
                  <td style={{ fontFamily: 'monospace', fontSize: '0.82rem' }}>{formatTime(log.check_time)}</td>
                  <td><strong style={{ color: 'var(--accent-info)' }}>{log.employee_id || '—'}</strong></td>
                  <td>{log.full_name || '—'}</td>
                  <td><span className="badge badge-info">{log.department || '—'}</span></td>
                  <td style={{ fontFamily: 'monospace' }}>{formatSimilarity(log.similarity)}</td>
                  <td>
                    <span className={`badge ${statusView.className}`} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      {statusView.icon} {statusView.label}
                    </span>
                  </td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem', maxWidth: 260 }}>{buildReason(log)}</td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{log.camera_id || '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {loading && <div className="empty-state"><div className="empty-state-icon">⏳</div><p>Đang tải nhật ký...</p></div>}
        {!loading && logs.length === 0 && <div className="empty-state"><div className="empty-state-icon">📋</div><p>Không có bản ghi chấm công</p></div>}
      </div>
      <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>Tổng: {logs.length} bản ghi</p>
    </div>
  );
}
