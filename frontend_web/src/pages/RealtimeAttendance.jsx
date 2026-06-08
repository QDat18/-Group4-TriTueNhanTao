import { useEffect, useState, useRef } from 'react';
import { RefreshCw, Play, Square, CheckCircle, Clock, AlertTriangle, Users, UserCheck, UserX, Percent, ShieldCheck } from 'lucide-react';
import { getAttendanceLogs, getReportSummary, getSettings } from '../api';

export default function RealtimeAttendance() {
  const [logs, setLogs] = useState([]);
  const [isStreaming, setIsStreaming] = useState(true);
  const [streamError, setStreamError] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState('system');
  const [currentFace, setCurrentFace] = useState(null);
  const [currentFaceTimestamp, setCurrentFaceTimestamp] = useState(null);
  const currentFacePollRef = useRef(null);
  const [stats, setStats] = useState({
    present: 0,
    late: 0,
    absent: 0,
    total: 0,
    rate: 0
  });
  const [policy, setPolicy] = useState({
    work_start_time: '08:00',
    allow_late_minutes: 30
  });
  const [streamUrl, setStreamUrl] = useState(() => {
    return `http://localhost:8000/api/attendance/stream?t=${Date.now()}`;
  });

  const pollIntervalRef = useRef(null);
  const BACKEND = 'http://localhost:8000';

  const loadData = () => {
    Promise.all([
      getAttendanceLogs({ limit: 15 }),
      getReportSummary('day')
    ])
      .then(([logsRes, summaryRes]) => {
        const fetchedLogs = logsRes.data || [];
        setLogs(fetchedLogs);

        const totalEmpCount = (summaryRes && !summaryRes.error) ? summaryRes.total_employees : 10;

        // Calculate daily stats dynamically based on logs of today (local date)
        const todayStr = new Date().toISOString().slice(0, 10);
        const todayLogs = fetchedLogs.filter(log => log.check_time && log.check_time.startsWith(todayStr));

        // Find unique employees present today
        const presentIds = new Set(todayLogs.map(log => log.employee_id));
        const lateLogs = todayLogs.filter(log => log.status === 'LATE');
        const uniqueLateIds = new Set(lateLogs.map(log => log.employee_id));

        const presentCount = presentIds.size;
        const lateCount = uniqueLateIds.size;
        const absentCount = Math.max(0, totalEmpCount - presentCount);
        const rate = totalEmpCount > 0 ? Math.round((presentCount / totalEmpCount) * 100) : 0;

        setStats({
          present: presentCount,
          late: lateCount,
          absent: absentCount,
          total: totalEmpCount,
          rate: rate
        });
      })
      .catch(err => {
        console.error("Lỗi đồng bộ dữ liệu:", err);
      });
  };

  useEffect(() => {
    // Fetch system policy config on load
    getSettings().then(res => {
      if (res && !res.error) {
        setPolicy({
          work_start_time: res.work_start_time || '08:00',
          allow_late_minutes: res.allow_late_minutes !== undefined ? res.allow_late_minutes : 30
        });
      }
    });
  }, []);

  useEffect(() => {
    if (isStreaming) {
      const timestamp = Date.now();
      const url = selectedCameraId === 'system'
        ? `http://localhost:8000/api/attendance/stream?t=${timestamp}`
        : `http://localhost:8000/api/attendance/stream?camera_id=${selectedCameraId}&t=${timestamp}`;
      setStreamUrl(url);
    } else {
      setStreamUrl('');
    }
  }, [isStreaming, selectedCameraId]);

  useEffect(() => {
    loadData();
    if (isStreaming) {
      pollIntervalRef.current = setInterval(() => {
        loadData();
      }, 3500);
    } else {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [isStreaming]);

  // Poll current face being scanned from the camera in real-time
  useEffect(() => {
    const pollCurrentFace = () => {
      fetch(`${BACKEND}/api/attendance/current-face`)
        .then(r => r.json())
        .then(res => {
          if (res.data) {
            setCurrentFace(res.data);
            setCurrentFaceTimestamp(Date.now());
          } else {
            // Clear display if no face detected for more than 3 seconds
            setCurrentFaceTimestamp(prev => {
              if (prev !== null && Date.now() - prev > 3000) {
                setCurrentFace(null);
                return null;
              }
              return prev;
            });
          }
        })
        .catch(() => {});
    };

    if (isStreaming) {
      pollCurrentFace();
      currentFacePollRef.current = setInterval(pollCurrentFace, 500);
    } else {
      clearInterval(currentFacePollRef.current);
      setCurrentFace(null);
      setCurrentFaceTimestamp(null);
    }

    return () => clearInterval(currentFacePollRef.current);
  }, [isStreaming]);

  const latestLog = logs.length > 0 ? logs[0] : null;
  const isRecent = latestLog ? (new Date() - new Date(latestLog.check_time)) < 8000 : false;
  // Display current face being scanned; fall back to latest log briefly after recognition
  const displayFace = currentFace || null;

  const formatTime = (isoString) => {
    if (!isoString) return '—';
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return (isoString || '').slice(11, 19);
    }
  };

  const formatDate = (isoString) => {
    if (!isoString) return '—';
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' });
    } catch {
      return (isoString || '').slice(0, 10);
    }
  };

  const getStatusDetails = (status) => {
    switch (status) {
      case 'SUCCESS':
        return {
          color: '#10b981',
          bg: '#ecfdf5',
          border: '#a7f3d0',
          text: 'Đúng giờ',
          icon: <CheckCircle size={14} style={{ color: '#10b981' }} />
        };
      case 'LATE':
        return {
          color: '#f59e0b',
          bg: '#fffbeb',
          border: '#fde68a',
          text: 'Đi muộn',
          icon: <Clock size={14} style={{ color: '#f59e0b' }} />
        };
      default:
        return {
          color: '#ef4444',
          bg: '#fef2f2',
          border: '#fca5a5',
          text: 'Lỗi / Trùng',
          icon: <AlertTriangle size={14} style={{ color: '#ef4444' }} />
        };
    }
  };

  const handleStreamError = () => {
    setIsStreaming(false);
    setStreamError(true);
  };

  const handleStartStream = () => {
    setStreamError(false);
    setIsStreaming(true);
  };

  return (
    <div className="animate-in" style={{ maxWidth: '1200px', margin: '0 auto', paddingBottom: '3rem' }}>

      {/* Page Title & Rules */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.02em', margin: 0 }}>
            Hệ thống Chấm công Live
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginTop: '0.2rem' }}>
            Nhận diện sinh trắc học thời gian thực, tự động tính toán thời gian đi muộn & chống giả mạo
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: '#f8fafc', border: '1px solid #e2e8f0', padding: '0.6rem 1rem', borderRadius: 'var(--radius-md)', fontSize: '0.82rem', color: '#334155', fontWeight: 600 }}>
          <ShieldCheck size={16} style={{ color: 'var(--accent-primary)' }} />
          <span>Giờ làm việc: <strong>{policy.work_start_time}</strong> (Cho phép trễ {policy.allow_late_minutes} phút)</span>
        </div>
      </div>

      {/* Real-time stats section */}
      <div className="grid-4" style={{ marginBottom: '1.5rem', gap: '1rem' }}>
        {/* Present Card */}
        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem' }}>
          <div style={{ width: 44, height: 44, borderRadius: 'var(--radius-md)', background: 'rgba(16, 185, 129, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#10b981' }}>
            <UserCheck size={22} />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>ĐÃ CÓ MẶT</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.1rem' }}>
              {stats.present} <span style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-muted)' }}>/ {stats.total} NV</span>
            </div>
          </div>
        </div>

        {/* Late Card */}
        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem' }}>
          <div style={{ width: 44, height: 44, borderRadius: 'var(--radius-md)', background: 'rgba(245, 158, 11, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#f59e0b' }}>
            <Clock size={22} />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>ĐI MUỘN</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.1rem' }}>
              {stats.late} <span style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-muted)' }}>nhân viên</span>
            </div>
          </div>
        </div>

        {/* Absent Card */}
        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem' }}>
          <div style={{ width: 44, height: 44, borderRadius: 'var(--radius-md)', background: 'rgba(239, 68, 68, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ef4444' }}>
            <UserX size={22} />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>VẮNG MẶT</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.1rem' }}>
              {stats.absent} <span style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-muted)' }}>chưa check-in</span>
            </div>
          </div>
        </div>

        {/* Attendance Rate Card */}
        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem' }}>
          <div style={{ width: 44, height: 44, borderRadius: 'var(--radius-md)', background: 'rgba(37, 99, 235, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-primary)' }}>
            <Percent size={20} />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>TỶ LỆ CHUYÊN CẦN</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.1rem' }}>
              {stats.rate}%
            </div>
          </div>
        </div>
      </div>

      {/* Main interactive grid */}
      <div className="grid-2" style={{ gap: '1.5rem', alignItems: 'stretch' }}>

        {/* Left: Camera Feed */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              📹 Camera giám sát (Cổng 01)
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
              <select
                value={selectedCameraId}
                onChange={e => {
                  setSelectedCameraId(e.target.value);
                  setStreamError(false);
                }}
                style={{
                  fontSize: '0.75rem',
                  padding: '0.25rem 0.5rem',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-color)',
                  background: 'var(--bg-card)',
                  color: 'var(--text-primary)',
                  fontWeight: 600,
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                <option value="system">Cấu hình hệ thống (Mặc định)</option>
                <option value="0">Thiết bị 0 (Webcam chính)</option>
                <option value="1">Thiết bị 1 (Webcam phụ)</option>
                <option value="2">Thiết bị 2</option>
                <option value="3">Thiết bị 3</option>
              </select>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', background: isStreaming ? '#ecfdf5' : '#f1f5f9', color: isStreaming ? '#059669' : '#64748b', padding: '0.25rem 0.65rem', borderRadius: '50px', fontSize: '0.72rem', fontWeight: 700 }}>
                <span className={`pulse-dot ${isStreaming ? 'active' : ''}`} style={{ backgroundColor: isStreaming ? '#10b981' : '#94a3b8' }} />
                {isStreaming ? 'LIVE STREAM' : 'OFFLINE'}
              </span>
            </div>
          </div>

          <div style={{
            background: '#090d16',
            borderRadius: 'var(--radius-md)',
            flexGrow: 1,
            minHeight: '380px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden',
            position: 'relative',
            border: '1px solid #1e293b',
            boxShadow: 'inset 0 4px 20px rgba(0,0,0,0.5)'
          }}>
            {isStreaming && streamUrl ? (
              <>
                <img
                  src={streamUrl}
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  alt="Camera Live Stream"
                  onError={handleStreamError}
                />

                {/* HUD Camera Target Indicator overlay */}
                <div className="camera-overlay">
                  <div className={`face-circle-guide ${isStreaming ? 'active' : ''}`}>
                    <div className="scanner-laser" />
                  </div>

                  {/* Info HUD */}
                  <div style={{ position: 'absolute', bottom: '1rem', left: '1rem', color: '#10b981', fontFamily: 'monospace', fontSize: '0.72rem', background: 'rgba(15,23,42,0.85)', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid #334155' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: '#22c55e', fontWeight: 'bold' }}>
                      <span style={{ display: 'inline-block', width: 6, height: 6, borderRadius: '50%', background: '#22c55e', animation: 'ping 1.2s infinite' }} />
                      <span>CAM_PORT_01: ONLINE</span>
                    </div>
                    <div style={{ color: '#94a3b8', marginTop: '0.2rem' }}>RESOLVER: ArcFace ResNet50</div>
                    <div style={{ color: '#94a3b8' }}>ANTI-SPOOFING: ACTIVE</div>
                  </div>
                </div>
              </>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', color: '#475569', textAlign: 'center', padding: '2rem' }}>
                <div style={{ fontSize: '3rem', opacity: 0.3 }}>📹</div>
                <h4 style={{ color: '#94a3b8', margin: 0, fontWeight: 600 }}>Thiết bị đang tạm dừng</h4>
                <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b', maxWidth: '280px' }}>
                  Bấm nút kích hoạt bên dưới để khởi chạy luồng quét camera thời gian thực.
                </p>
                {streamError && (
                  <p style={{ color: 'var(--accent-danger)', fontSize: '0.78rem', marginTop: '0.5rem', fontWeight: 500 }}>
                    Lỗi kết nối camera. Vui lòng kiểm tra quyền truy cập webcam trên trình duyệt hoặc uvicorn backend.
                  </p>
                )}
              </div>
            )}
          </div>

          <div style={{ marginTop: '1rem' }}>
            {isStreaming ? (
              <button
                className="btn btn-danger"
                style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '0.75rem' }}
                onClick={() => setIsStreaming(false)}
              >
                <Square size={14} /> Dừng Quét Điểm Danh
              </button>
            ) : (
              <button
                className="btn btn-primary"
                style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '0.75rem' }}
                onClick={handleStartStream}
              >
                <Play size={14} /> Bắt đầu Quét Điểm Danh
              </button>
            )}
          </div>
        </div>

        {/* Right: Recognized Employee details */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem', marginBottom: '1.25rem' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              👤 Nhân viên vừa ghi nhận
            </span>
          </div>

          {displayFace ? (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              flexGrow: 1,
              animation: 'pulse-green 1.5s ease-out'
            }}>

              {/* Profile Avatar Ring */}
              <div style={{
                width: 140,
                height: 140,
                borderRadius: '50%',
                overflow: 'hidden',
                border: `4px solid ${
                  displayFace.status === 'SUCCESS' ? '#10b981'
                  : displayFace.status === 'LATE' ? '#f59e0b'
                  : displayFace.status === 'COOLDOWN' ? '#6366f1'
                  : '#ef4444'
                }`,
                marginBottom: '1rem',
                background: '#f8fafc',
                boxShadow: '0 10px 25px rgba(0,0,0,0.06)',
                position: 'relative'
              }}>
                {/* Look up employee_id from logs by name match */}
                {(() => {
                  const matched = logs.find(l => l.full_name === displayFace.full_name);
                  const empId = matched ? matched.employee_id : null;
                  return (
                    <img
                      src={empId
                        ? `${BACKEND}/api/portraits/${empId}/${empId}_000.jpg`
                        : `https://ui-avatars.com/api/?name=${encodeURIComponent(displayFace.full_name)}&background=2563eb&color=fff&size=200`
                      }
                      alt={displayFace.full_name}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      onError={(e) => {
                        e.target.onerror = null;
                        e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(displayFace.full_name)}&background=2563eb&color=fff&size=200`;
                      }}
                    />
                  );
                })()}
              </div>

              {/* Name and Basic details */}
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-primary)', margin: '0 0 0.25rem 0', textAlign: 'center' }}>
                {displayFace.full_name}
              </h2>
              {(() => {
                const matched = logs.find(l => l.full_name === displayFace.full_name);
                return matched ? (
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 500, margin: '0 0 1rem 0' }}>
                    ID: <strong style={{ color: 'var(--text-primary)' }}>{matched.employee_id}</strong> &nbsp;•&nbsp; Phòng: <strong style={{ color: 'var(--text-primary)' }}>{matched.department || '—'}</strong>
                  </p>
                ) : (
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 500, margin: '0 0 1rem 0' }}>
                    Đang nhận diện...
                  </p>
                );
              })()}

              {/* Status Pill */}
              {(() => {
                const statusMap = {
                  SUCCESS: { color: '#10b981', bg: '#ecfdf5', border: '#a7f3d0', text: 'Đúng giờ', icon: <CheckCircle size={14} style={{ color: '#10b981' }} /> },
                  LATE:    { color: '#f59e0b', bg: '#fffbeb', border: '#fde68a', text: 'Đi muộn',  icon: <Clock size={14} style={{ color: '#f59e0b' }} /> },
                  COOLDOWN:{ color: '#6366f1', bg: '#eef2ff', border: '#c7d2fe', text: 'Trong cooldown', icon: <Clock size={14} style={{ color: '#6366f1' }} /> },
                };
                const s = statusMap[displayFace.status] || { color: '#ef4444', bg: '#fef2f2', border: '#fca5a5', text: 'Đang quét...', icon: <AlertTriangle size={14} style={{ color: '#ef4444' }} /> };
                return (
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: '0.35rem',
                    background: s.bg, color: s.color, border: `1px solid ${s.border}`,
                    fontSize: '0.82rem', fontWeight: 700,
                    padding: '0.45rem 1.25rem', borderRadius: '50px',
                    marginBottom: '1.5rem', boxShadow: '0 2px 6px rgba(0,0,0,0.02)'
                  }}>
                    {s.icon}<span>{s.text.toUpperCase()}</span>
                  </div>
                );
              })()}

              {/* Liveness score */}
              <div style={{
                width: '100%', background: '#f8fafc',
                padding: '1.25rem', borderRadius: 'var(--radius-md)',
                border: '1px solid #e2e8f0'
              }}>
                <div>
                  <span style={{ display: 'block', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.05em', marginBottom: '0.25rem', textTransform: 'uppercase' }}>
                    Liveness Score (Anti-Spoofing)
                  </span>
                  <span style={{ fontWeight: 800, fontSize: '1.2rem', fontFamily: 'monospace', color: displayFace.liveness_score > 0.6 ? '#10b981' : '#f59e0b' }}>
                    {displayFace.liveness_score ? `${(displayFace.liveness_score * 100).toFixed(1)}%` : '—'}
                  </span>
                </div>
                <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '0.75rem', marginTop: '0.75rem' }}>
                  <span style={{ display: 'block', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.05em', marginBottom: '0.25rem', textTransform: 'uppercase' }}>
                    Trạng thái nhận diện
                  </span>
                  <span style={{ fontWeight: 700, fontSize: '0.92rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                    {displayFace.label}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', padding: '5rem 0', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flexGrow: 1 }}>
              <div style={{ fontSize: '3.5rem', marginBottom: '1rem', opacity: 0.3 }}>👤</div>
              <p style={{ margin: 0, fontSize: '0.88rem', fontWeight: 500 }}>Đang đợi ghi nhận check-in...</p>
              <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.78rem', color: 'var(--text-muted)' }}>Vui lòng đứng trực diện camera góc rộng.</p>
            </div>
          )}
        </div>
      </div>

      {/* Bottom section: Log Table */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem', marginBottom: '1rem' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            📋 Nhật ký check-in hôm nay
          </span>
          <button className="btn btn-sm btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }} onClick={loadData}>
            <RefreshCw size={12} /> Làm mới bảng
          </button>
        </div>

        {logs.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: 50, textAlign: 'center' }}>Ảnh</th>
                  <th>Thời gian</th>
                  <th>Ngày</th>
                  <th>Mã Nhân Viên</th>
                  <th>Họ tên</th>
                  <th>Phòng ban</th>
                  <th style={{ textAlign: 'center' }}>Mẫu khớp</th>
                  <th style={{ textAlign: 'center' }}>Trạng thái</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log, i) => {
                  const statusDetails = getStatusDetails(log.status);
                  return (
                    <tr key={i} style={i === 0 && isRecent ? { background: 'rgba(16, 185, 129, 0.06)', transition: 'background 0.5s ease' } : {}}>
                      <td style={{ textAlign: 'center' }}>
                        <div style={{ width: 32, height: 32, borderRadius: '50%', overflow: 'hidden', border: '1px solid var(--border-color)', background: '#f1f5f9', margin: '0 auto' }}>
                          <img
                            src={`http://localhost:8000/api/portraits/${log.employee_id}/${log.employee_id}_000.jpg`}
                            alt={log.full_name}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                            onError={(e) => {
                              e.target.onerror = null;
                              e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(log.full_name || 'User')}&background=random&color=fff&size=80`;
                            }}
                          />
                        </div>
                      </td>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.85rem', fontWeight: 700 }}>
                        {formatTime(log.check_time)}
                      </td>
                      <td style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                        {formatDate(log.check_time)}
                      </td>
                      <td>
                        <strong style={{ color: 'var(--accent-primary)', fontSize: '0.85rem' }}>{log.employee_id}</strong>
                      </td>
                      <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {log.full_name || '—'}
                      </td>
                      <td style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                        {log.department || '—'}
                      </td>
                      <td style={{ fontFamily: 'monospace', fontWeight: 700, color: 'var(--accent-primary)', textAlign: 'center' }}>
                        {log.similarity ? `${Math.round(log.similarity * 100)}%` : '—'}
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.25rem',
                          background: statusDetails.bg,
                          color: statusDetails.color,
                          border: `1px solid ${statusDetails.border}`,
                          fontSize: '0.72rem',
                          fontWeight: 700,
                          padding: '0.2rem 0.65rem',
                          borderRadius: '20px'
                        }}>
                          {statusDetails.text.toUpperCase()}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state" style={{ padding: '2rem' }}>
            <Users size={32} style={{ opacity: 0.3, marginBottom: '0.5rem' }} />
            <p style={{ margin: 0 }}>Chưa có log chấm công được ghi nhận trong hôm nay</p>
          </div>
        )}
      </div>
    </div>
  );
}
