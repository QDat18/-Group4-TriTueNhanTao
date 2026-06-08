import { useEffect, useState, useRef } from 'react';
import {
  RefreshCw, Play, Square, CheckCircle, Clock, AlertTriangle,
  Users, UserCheck, UserX, Percent, ShieldCheck, LogIn, LogOut,
  TrendingUp, TrendingDown, Minus, Eye, EyeOff, Wifi, WifiOff
} from 'lucide-react';
import { getAttendanceLogs, getReportSummary, getSettings } from '../api';

// ── Trạng thái chấm công thực tế ──────────────────────────────────────────────
// CHECK_IN  + SUCCESS  → Vào đúng giờ
// CHECK_IN  + LATE     → Vào muộn
// CHECK_OUT + SUCCESS  → Về đúng giờ
// CHECK_OUT + EARLY    → Về sớm
// SCANNING             → Đang nhận diện (chưa rõ)
// UNKNOWN / LOW_CONF   → Không nhận ra / Độ tin cậy thấp
// COOLDOWN             → Trùng lặp / Cooldown

const STATUS_CONFIG = {
  // Check-in statuses
  SUCCESS: { color: '#10b981', bg: '#ecfdf5', border: '#6ee7b7', text: 'Vào đúng giờ', icon: 'check-in', type: 'in' },
  LATE: { color: '#f59e0b', bg: '#fffbeb', border: '#fcd34d', text: 'Vào muộn', icon: 'late', type: 'in' },
  // Check-out statuses
  CHECK_OUT: { color: '#3b82f6', bg: '#eff6ff', border: '#93c5fd', text: 'Ra đúng giờ', icon: 'check-out', type: 'out' },
  EARLY_LEAVE: { color: '#f97316', bg: '#fff7ed', border: '#fdba74', text: 'Về sớm', icon: 'early', type: 'out' },
  // Uncertain
  SCANNING: { color: '#8b5cf6', bg: '#f5f3ff', border: '#c4b5fd', text: 'Đang nhận diện', icon: 'scan', type: 'scan' },
  UNKNOWN: { color: '#ef4444', bg: '#fef2f2', border: '#fca5a5', text: 'Không nhận ra', icon: 'unknown', type: 'err' },
  LOW_CONF: { color: '#f97316', bg: '#fff7ed', border: '#fdba74', text: 'Độ tin cậy thấp', icon: 'low', type: 'warn' },
  COOLDOWN: { color: '#6366f1', bg: '#eef2ff', border: '#a5b4fc', text: 'Đã điểm danh', icon: 'cooldown', type: 'dup' },
  SPOOFING: { color: '#dc2626', bg: '#fef2f2', border: '#f87171', text: 'Phát hiện gian lận', icon: 'spoof', type: 'err' },
};

function StatusIcon({ status, size = 14 }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.UNKNOWN;
  switch (cfg.icon) {
    case 'check-in': return <LogIn size={size} color={cfg.color} />;
    case 'check-out': return <LogOut size={size} color={cfg.color} />;
    case 'late': return <Clock size={size} color={cfg.color} />;
    case 'early': return <TrendingDown size={size} color={cfg.color} />;
    case 'scan': return <Eye size={size} color={cfg.color} />;
    case 'unknown': return <EyeOff size={size} color={cfg.color} />;
    case 'low': return <AlertTriangle size={size} color={cfg.color} />;
    case 'cooldown': return <Minus size={size} color={cfg.color} />;
    case 'spoof': return <ShieldCheck size={size} color={cfg.color} />;
    default: return <AlertTriangle size={size} color={cfg.color} />;
  }
}

function StatusBadge({ status, size = 'sm' }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.UNKNOWN;
  const pad = size === 'lg' ? '0.5rem 1.1rem' : '0.22rem 0.6rem';
  const fs = size === 'lg' ? '0.82rem' : '0.7rem';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '0.3rem',
      background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}`,
      fontSize: fs, fontWeight: 700, padding: pad, borderRadius: '20px',
      letterSpacing: '0.03em'
    }}>
      <StatusIcon status={status} size={size === 'lg' ? 13 : 11} />
      {cfg.text.toUpperCase()}
    </span>
  );
}

function ConfidenceBar({ value }) {
  const pct = value ? Math.round(value * 100) : 0;
  const color = pct >= 85 ? '#10b981' : pct >= 70 ? '#f59e0b' : '#ef4444';
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
        <span style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Độ khớp khuôn mặt</span>
        <span style={{ fontSize: '0.75rem', fontWeight: 800, fontFamily: 'monospace', color }}>{pct}%</span>
      </div>
      <div style={{ height: 6, borderRadius: 99, background: '#e2e8f0', overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, borderRadius: 99,
          background: `linear-gradient(90deg, ${color}aa, ${color})`,
          transition: 'width 0.6s ease'
        }} />
      </div>
      {pct < 70 && (
        <p style={{ margin: '0.3rem 0 0', fontSize: '0.68rem', color: '#ef4444', fontWeight: 600 }}>
          ⚠ Độ tin cậy thấp — cần xác nhận thủ công
        </p>
      )}
    </div>
  );
}

function LivenessBar({ value }) {
  const pct = value ? Math.round(value * 100) : 0;
  const color = pct >= 60 ? '#10b981' : '#ef4444';
  const label = pct >= 60 ? 'Thật' : 'Nghi ngờ giả mạo';
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
        <span style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Anti-Spoofing (Liveness)</span>
        <span style={{ fontSize: '0.75rem', fontWeight: 800, fontFamily: 'monospace', color }}>
          {pct}% — {label}
        </span>
      </div>
      <div style={{ height: 6, borderRadius: 99, background: '#e2e8f0', overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, borderRadius: 99,
          background: `linear-gradient(90deg, ${color}aa, ${color})`,
          transition: 'width 0.6s ease'
        }} />
      </div>
    </div>
  );
}

// ── Portrait thumbnail ─────────────────────────────────────────────────────────
function Portrait({ empId, name, size = 40, backend = 'http://localhost:8000' }) {
  return (
    <img
      src={empId
        ? `${backend}/api/portraits/${empId}/${empId}_000.jpg`
        : `https://ui-avatars.com/api/?name=${encodeURIComponent(name || 'NV')}&background=6366f1&color=fff&size=200`}
      alt={name}
      style={{ width: size, height: size, objectFit: 'cover', borderRadius: size >= 80 ? '50%' : '50%' }}
      onError={e => {
        e.target.onerror = null;
        e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(name || 'NV')}&background=6366f1&color=fff&size=200`;
      }}
    />
  );
}

export default function RealtimeAttendance() {
  const BACKEND = 'http://localhost:8000';

  const [logs, setLogs] = useState([]);
  const [isStreaming, setIsStreaming] = useState(true);
  const [streamError, setStreamError] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState('system');
  const [currentFace, setCurrentFace] = useState(null);
  const [currentFaceTs, setCurrentFaceTs] = useState(null);
  const [stats, setStats] = useState({ present: 0, late: 0, earlyLeave: 0, absent: 0, total: 0, rate: 0 });
  const [policy, setPolicy] = useState({ work_start_time: '08:00', work_end_time: '17:30', allow_late_minutes: 15, allow_early_minutes: 15 });
  const [streamUrl, setStreamUrl] = useState(`${BACKEND}/api/attendance/stream?t=${Date.now()}`);
  const [now, setNow] = useState(new Date());

  const pollIntervalRef = useRef(null);
  const facePollRef = useRef(null);
  const clockRef = useRef(null);

  // Live clock
  useEffect(() => {
    clockRef.current = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(clockRef.current);
  }, []);

  // Load policy
  useEffect(() => {
    getSettings().then(res => {
      if (res && !res.error) {
        setPolicy(p => ({
          ...p,
          work_start_time: res.work_start_time || '08:00',
          work_end_time: res.work_end_time || '17:30',
          allow_late_minutes: res.allow_late_minutes ?? 15,
          allow_early_minutes: res.allow_early_minutes ?? 15,
        }));
      }
    }).catch(() => { });
  }, []);

  // Load logs + stats
  const loadData = () => {
    Promise.all([
      getAttendanceLogs({ limit: 20 }),
      getReportSummary('day'),
    ]).then(([logsRes, summaryRes]) => {
      const fetched = logsRes.data || [];
      setLogs(fetched);

      const totalEmp = (summaryRes && !summaryRes.error) ? (summaryRes.total_employees || 10) : 10;
      const today = new Date().toISOString().slice(0, 10);
      const todayLogs = fetched.filter(l => l.check_time?.startsWith(today));

      const presentIds = new Set(todayLogs.filter(l => ['SUCCESS', 'LATE', 'CHECK_OUT', 'EARLY_LEAVE'].includes(l.status)).map(l => l.employee_id));
      const lateIds = new Set(todayLogs.filter(l => l.status === 'LATE').map(l => l.employee_id));
      const earlyIds = new Set(todayLogs.filter(l => l.status === 'EARLY_LEAVE').map(l => l.employee_id));
      const present = presentIds.size;
      const absent = Math.max(0, totalEmp - present);
      const rate = totalEmp > 0 ? Math.round((present / totalEmp) * 100) : 0;

      setStats({ present, late: lateIds.size, earlyLeave: earlyIds.size, absent, total: totalEmp, rate });
    }).catch(console.error);
  };

  // Streaming URL
  useEffect(() => {
    if (isStreaming) {
      const t = Date.now();
      const cam = selectedCameraId === 'system' ? '' : `&camera_id=${selectedCameraId}`;
      setStreamUrl(`${BACKEND}/api/attendance/stream?t=${t}${cam}`);
    } else {
      setStreamUrl('');
    }
  }, [isStreaming, selectedCameraId]);

  // Polling logs
  useEffect(() => {
    loadData();
    if (isStreaming) {
      pollIntervalRef.current = setInterval(loadData, 3500);
    } else {
      clearInterval(pollIntervalRef.current);
    }
    return () => clearInterval(pollIntervalRef.current);
  }, [isStreaming]);

  // Poll current face
  useEffect(() => {
    const poll = () => {
      fetch(`${BACKEND}/api/attendance/current-face`)
        .then(r => r.json())
        .then(res => {
          if (res.data) {
            setCurrentFace(res.data);
            setCurrentFaceTs(Date.now());
          } else {
            setCurrentFaceTs(prev => {
              if (prev !== null && Date.now() - prev > 3000) { setCurrentFace(null); return null; }
              return prev;
            });
          }
        }).catch(() => { });
    };
    if (isStreaming) { poll(); facePollRef.current = setInterval(poll, 500); }
    else { clearInterval(facePollRef.current); setCurrentFace(null); setCurrentFaceTs(null); }
    return () => clearInterval(facePollRef.current);
  }, [isStreaming]);

  const formatTime = (iso) => {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }); }
    catch { return iso.slice(11, 19); }
  };
  const formatDate = (iso) => {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' }); }
    catch { return iso.slice(0, 10); }
  };

  const latestLog = logs[0] || null;
  const isRecent = latestLog ? (new Date() - new Date(latestLog.check_time)) < 8000 : false;

  // Current face data — resolve employee_id from logs by name match
  const faceEmpId = currentFace
    ? (logs.find(l => l.full_name === currentFace.full_name)?.employee_id || null)
    : null;

  // Determine face ring color
  const ringColor = currentFace
    ? (STATUS_CONFIG[currentFace.status]?.color || '#94a3b8')
    : '#334155';

  // Current time display
  const timeStr = now.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const dateStr = now.toLocaleDateString('vi-VN', { weekday: 'long', day: '2-digit', month: '2-digit', year: 'numeric' });

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="animate-in" style={{ maxWidth: 1280, margin: '0 auto', paddingBottom: '3rem' }}>

      {/* ── Header ── */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
        marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem'
      }}>
        <div>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.02em', margin: 0 }}>
            Chấm công thời gian thực
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
            Nhận diện khuôn mặt · Ghi nhận vào / ra · Chống gian lận (Liveness)
          </p>
        </div>

        {/* Policy + clock */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.4rem' }}>
          <div style={{ fontFamily: 'monospace', fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '0.04em' }}>
            {timeStr}
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{dateStr}</div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: '0.45rem',
            background: '#f8fafc', border: '1px solid #e2e8f0',
            padding: '0.4rem 0.85rem', borderRadius: '8px',
            fontSize: '0.78rem', color: '#334155', fontWeight: 600
          }}>
            <ShieldCheck size={13} color="var(--accent-primary)" />
            Vào: <strong>{policy.work_start_time}</strong>
            &nbsp;·&nbsp;
            Ra: <strong>{policy.work_end_time}</strong>
            &nbsp;·&nbsp;
            Trễ tối đa: <strong>{policy.allow_late_minutes} phút</strong>
            &nbsp;·&nbsp;
            Về sớm tối đa: <strong>{policy.allow_early_minutes} phút</strong>
          </div>
        </div>
      </div>

      {/* ── Stats row ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        {[
          { label: 'Đã vào', value: stats.present, sub: `/ ${stats.total} NV`, color: '#10b981', Icon: UserCheck },
          { label: 'Vào muộn', value: stats.late, sub: 'nhân viên', color: '#f59e0b', Icon: Clock },
          { label: 'Về sớm', value: stats.earlyLeave, sub: 'nhân viên', color: '#f97316', Icon: TrendingDown },
          { label: 'Vắng mặt', value: stats.absent, sub: 'chưa check-in', color: '#ef4444', Icon: UserX },
          { label: 'Chuyên cần', value: `${stats.rate}%`, sub: 'hôm nay', color: '#3b82f6', Icon: Percent },
        ].map(({ label, value, sub, color, Icon }) => (
          <div key={label} className="card" style={{ display: 'flex', alignItems: 'center', gap: '0.9rem', padding: '1.1rem' }}>
            <div style={{ width: 42, height: 42, borderRadius: '10px', background: `${color}18`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Icon size={20} color={color} />
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</div>
              <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-primary)', lineHeight: 1.2 }}>
                {value} <span style={{ fontSize: '0.78rem', fontWeight: 500, color: 'var(--text-muted)' }}>{sub}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* ── Main 2-col ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: '1.5rem', alignItems: 'start' }}>

        {/* Left: Camera */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              📹 Camera nhận diện
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <select
                value={selectedCameraId}
                onChange={e => { setSelectedCameraId(e.target.value); setStreamError(false); }}
                style={{
                  fontSize: '0.75rem', padding: '0.22rem 0.5rem',
                  borderRadius: '6px', border: '1px solid var(--border-color)',
                  background: 'var(--bg-card)', color: 'var(--text-primary)', fontWeight: 600
                }}
              >
                <option value="system">Hệ thống (mặc định)</option>
                <option value="0">Webcam 0 (chính)</option>
                <option value="1">Webcam 1 (phụ)</option>
                <option value="2">Webcam 2</option>
              </select>
              <span style={{
                display: 'flex', alignItems: 'center', gap: '0.3rem',
                background: isStreaming ? '#ecfdf5' : '#f1f5f9',
                color: isStreaming ? '#059669' : '#64748b',
                padding: '0.2rem 0.6rem', borderRadius: '50px', fontSize: '0.7rem', fontWeight: 700
              }}>
                {isStreaming ? <Wifi size={11} /> : <WifiOff size={11} />}
                {isStreaming ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>
          </div>

          {/* Camera viewport */}
          <div style={{
            background: '#090d16', borderRadius: '10px', minHeight: 400,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            overflow: 'hidden', position: 'relative',
            border: '1px solid #1e293b', boxShadow: 'inset 0 4px 20px rgba(0,0,0,0.5)'
          }}>
            {isStreaming && streamUrl ? (
              <>
                <img src={streamUrl} style={{ width: '100%', height: '100%', objectFit: 'contain' }} alt="Live Stream" onError={() => { setIsStreaming(false); setStreamError(true); }} />

                {/* HUD overlay */}
                <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
                  {/* Scanner ring */}
                  <div className="camera-overlay">
                    <div className={`face-circle-guide ${isStreaming ? 'active' : ''}`}>
                      <div className="scanner-laser" />
                    </div>
                  </div>
                  {/* Bottom-left info */}
                  <div style={{
                    position: 'absolute', bottom: '0.85rem', left: '0.85rem',
                    fontFamily: 'monospace', fontSize: '0.68rem',
                    background: 'rgba(9,13,22,0.88)', padding: '0.5rem 0.75rem',
                    borderRadius: '6px', border: '1px solid #334155', color: '#94a3b8'
                  }}>
                    <div style={{ color: '#22c55e', fontWeight: 700, marginBottom: '0.15rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', display: 'inline-block', animation: 'ping 1.2s infinite' }} />
                      CAM ONLINE · ArcFace ResNet50
                    </div>
                    <div>Anti-Spoofing: ACTIVE</div>
                    <div>Threshold: 85% match · Liveness: 60%</div>
                  </div>

                  {/* Last recognition flash */}
                  {isRecent && latestLog && (
                    <div style={{
                      position: 'absolute', top: '0.75rem', right: '0.75rem',
                      background: 'rgba(9,13,22,0.88)', border: `1px solid ${STATUS_CONFIG[latestLog.status]?.border || '#334155'}`,
                      padding: '0.4rem 0.7rem', borderRadius: '8px', fontSize: '0.72rem',
                      fontFamily: 'monospace', color: STATUS_CONFIG[latestLog.status]?.color || '#fff'
                    }}>
                      ✓ {latestLog.full_name} — {STATUS_CONFIG[latestLog.status]?.text || latestLog.status}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.85rem', color: '#475569', textAlign: 'center', padding: '2rem' }}>
                <div style={{ fontSize: '2.5rem', opacity: 0.25 }}>📹</div>
                <p style={{ margin: 0, fontWeight: 600, color: '#94a3b8' }}>Camera tạm dừng</p>
                {streamError && <p style={{ color: '#ef4444', fontSize: '0.78rem', fontWeight: 500, margin: 0 }}>Lỗi kết nối camera. Kiểm tra backend / quyền webcam.</p>}
              </div>
            )}
          </div>

          {/* Control button */}
          <div style={{ marginTop: '0.85rem' }}>
            {isStreaming ? (
              <button
                className="btn btn-danger"
                style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
                onClick={() => setIsStreaming(false)}
              >
                <Square size={13} /> Dừng quét
              </button>
            ) : (
              <button
                className="btn btn-primary"
                style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
                onClick={() => { setStreamError(false); setIsStreaming(true); }}
              >
                <Play size={13} /> Bắt đầu quét
              </button>
            )}
          </div>
        </div>

        {/* Right: Recognition panel */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.65rem', marginBottom: '1rem' }}>
            👤 Nhận diện gần nhất
          </div>

          {currentFace ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>

              {/* Portrait ring */}
              <div style={{
                width: 120, height: 120, borderRadius: '50%', overflow: 'hidden',
                border: `4px solid ${ringColor}`,
                boxShadow: `0 0 20px ${ringColor}44`,
                background: '#f8fafc', flexShrink: 0
              }}>
                <Portrait empId={faceEmpId} name={currentFace.full_name} size={120} backend={BACKEND} />
              </div>

              {/* Name */}
              <div style={{ textAlign: 'center' }}>
                <h2 style={{ fontSize: '1.3rem', fontWeight: 800, margin: '0 0 0.15rem', color: 'var(--text-primary)' }}>
                  {currentFace.full_name || 'Đang nhận diện...'}
                </h2>
                {faceEmpId && (
                  <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                    ID: <strong>{faceEmpId}</strong>
                    {(logs.find(l => l.employee_id === faceEmpId)?.department) &&
                      <> · Phòng: <strong>{logs.find(l => l.employee_id === faceEmpId).department}</strong></>
                    }
                  </p>
                )}
              </div>

              {/* Status badge */}
              <StatusBadge status={currentFace.status} size="lg" />

              {/* Time note */}
              {currentFace.status === 'LATE' && (
                <div style={{ background: '#fffbeb', border: '1px solid #fcd34d', borderRadius: '8px', padding: '0.5rem 0.85rem', fontSize: '0.78rem', color: '#92400e', fontWeight: 600, textAlign: 'center' }}>
                  ⏰ Muộn so với giờ quy định ({policy.work_start_time})
                </div>
              )}
              {currentFace.status === 'EARLY_LEAVE' && (
                <div style={{ background: '#fff7ed', border: '1px solid #fdba74', borderRadius: '8px', padding: '0.5rem 0.85rem', fontSize: '0.78rem', color: '#9a3412', fontWeight: 600, textAlign: 'center' }}>
                  🏃 Về trước giờ tan làm ({policy.work_end_time})
                </div>
              )}
              {currentFace.status === 'UNKNOWN' && (
                <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: '8px', padding: '0.5rem 0.85rem', fontSize: '0.78rem', color: '#991b1b', fontWeight: 600, textAlign: 'center' }}>
                  ❌ Không tìm thấy trong hệ thống — cần đăng ký
                </div>
              )}
              {currentFace.status === 'SCANNING' && (
                <div style={{ background: '#f5f3ff', border: '1px solid #c4b5fd', borderRadius: '8px', padding: '0.5rem 0.85rem', fontSize: '0.78rem', color: '#5b21b6', fontWeight: 600, textAlign: 'center' }}>
                  🔍 Đang so khớp khuôn mặt...
                </div>
              )}

              {/* Metrics */}
              <div style={{ width: '100%', background: '#f8fafc', borderRadius: '10px', padding: '0.9rem', border: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <ConfidenceBar value={currentFace.similarity} />
                <LivenessBar value={currentFace.liveness_score} />

                {currentFace.label && (
                  <div>
                    <span style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 600, display: 'block', marginBottom: '0.2rem' }}>NHÃN NHẬN DIỆN</span>
                    <span style={{ fontFamily: 'monospace', fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)' }}>{currentFace.label}</span>
                  </div>
                )}

                {currentFace.check_time && (
                  <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '0.65rem' }}>
                    <span style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 600, display: 'block', marginBottom: '0.2rem' }}>THỜI GIAN GHI NHẬN</span>
                    <span style={{ fontFamily: 'monospace', fontSize: '0.9rem', fontWeight: 800, color: 'var(--text-primary)' }}>{formatTime(currentFace.check_time)}</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: '0.5rem' }}>{formatDate(currentFace.check_time)}</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flexGrow: 1, color: 'var(--text-muted)', padding: '3rem 1rem', gap: '0.75rem' }}>
              <div style={{ fontSize: '2.8rem', opacity: 0.2 }}>👤</div>
              <p style={{ margin: 0, fontWeight: 600, fontSize: '0.85rem' }}>Đang chờ nhận diện...</p>
              <p style={{ margin: 0, fontSize: '0.75rem', textAlign: 'center', maxWidth: '200px' }}>Vui lòng đứng trực diện camera. Tránh đội mũ, đeo khẩu trang.</p>
            </div>
          )}
        </div>
      </div>

      {/* ── Log table ── */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.65rem', marginBottom: '1rem' }}>
          <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            📋 Nhật ký chấm công hôm nay
          </span>
          <button
            className="btn btn-sm btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}
            onClick={loadData}
          >
            <RefreshCw size={11} /> Làm mới
          </button>
        </div>

        {logs.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: 44, textAlign: 'center' }}>Ảnh</th>
                  <th>Thời gian</th>
                  <th>Ngày</th>
                  <th>Mã NV</th>
                  <th>Họ tên</th>
                  <th>Phòng ban</th>
                  <th style={{ textAlign: 'center' }}>Loại</th>
                  <th style={{ textAlign: 'center' }}>Độ khớp</th>
                  <th style={{ textAlign: 'center' }}>Liveness</th>
                  <th style={{ textAlign: 'center' }}>Trạng thái</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log, i) => {
                  const cfg = STATUS_CONFIG[log.status] || STATUS_CONFIG.UNKNOWN;
                  const sim = log.similarity ? Math.round(log.similarity * 100) : null;
                  const liv = log.liveness_score ? Math.round(log.liveness_score * 100) : null;
                  const isCheckOut = ['CHECK_OUT', 'EARLY_LEAVE'].includes(log.status);
                  return (
                    <tr key={i} style={i === 0 && isRecent ? { background: `${cfg.bg}` } : {}}>
                      {/* Portrait */}
                      <td style={{ textAlign: 'center', padding: '0.5rem' }}>
                        <div style={{
                          width: 34, height: 34, borderRadius: '50%',
                          overflow: 'hidden', border: `2px solid ${cfg.border}`,
                          background: '#f1f5f9', margin: '0 auto'
                        }}>
                          <Portrait empId={log.employee_id} name={log.full_name} size={34} backend={BACKEND} />
                        </div>
                      </td>
                      {/* Time */}
                      <td style={{ fontFamily: 'monospace', fontSize: '0.83rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                        {formatTime(log.check_time)}
                      </td>
                      {/* Date */}
                      <td style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                        {formatDate(log.check_time)}
                      </td>
                      {/* Employee ID */}
                      <td>
                        <strong style={{ color: 'var(--accent-primary)', fontSize: '0.82rem' }}>{log.employee_id}</strong>
                      </td>
                      {/* Name */}
                      <td style={{ fontWeight: 600, fontSize: '0.88rem' }}>{log.full_name || '—'}</td>
                      {/* Department */}
                      <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{log.department || '—'}</td>
                      {/* Check-in / out */}
                      <td style={{ textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-flex', alignItems: 'center', gap: '0.25rem',
                          fontSize: '0.7rem', fontWeight: 700,
                          color: isCheckOut ? '#3b82f6' : '#10b981'
                        }}>
                          {isCheckOut ? <LogOut size={11} /> : <LogIn size={11} />}
                          {isCheckOut ? 'Ra' : 'Vào'}
                        </span>
                      </td>
                      {/* Similarity */}
                      <td style={{ textAlign: 'center' }}>
                        {sim !== null ? (
                          <span style={{
                            fontFamily: 'monospace', fontWeight: 700, fontSize: '0.8rem',
                            color: sim >= 85 ? '#10b981' : sim >= 70 ? '#f59e0b' : '#ef4444'
                          }}>
                            {sim}%
                          </span>
                        ) : '—'}
                      </td>
                      {/* Liveness */}
                      <td style={{ textAlign: 'center' }}>
                        {liv !== null ? (
                          <span style={{
                            fontFamily: 'monospace', fontWeight: 700, fontSize: '0.8rem',
                            color: liv >= 60 ? '#10b981' : '#ef4444'
                          }}>
                            {liv}%
                          </span>
                        ) : '—'}
                      </td>
                      {/* Status badge */}
                      <td style={{ textAlign: 'center' }}>
                        <StatusBadge status={log.status} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state" style={{ padding: '2.5rem' }}>
            <Users size={32} style={{ opacity: 0.25, marginBottom: '0.5rem' }} />
            <p style={{ margin: 0 }}>Chưa có log chấm công trong hôm nay</p>
          </div>
        )}
      </div>

    </div>
  );
}