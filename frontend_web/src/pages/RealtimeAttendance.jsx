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
  ABSENT: { color: '#ef4444', bg: '#fef2f2', border: '#fca5a5', text: 'Vắng mặt', icon: 'unknown', type: 'err' },
  MISSING_CHECK_OUT: { color: '#f59e0b', bg: '#fffbeb', border: '#fcd34d', text: 'Quên check-out', icon: 'late', type: 'in' },
  REJECTED_CHECK_IN: { color: '#ef4444', bg: '#fef2f2', border: '#fca5a5', text: 'Từ chối check-in (Quá giờ)', icon: 'unknown', type: 'err' },
  REJECTED_CHECK_OUT: { color: '#ef4444', bg: '#fef2f2', border: '#fca5a5', text: 'Từ chối check-out (Chưa đến giờ)', icon: 'unknown', type: 'err' },
  COMPLETED: { color: '#10b981', bg: '#ecfdf5', border: '#6ee7b7', text: 'Hoàn tất chấm công', icon: 'check-in', type: 'in' },
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
function getLocalDateKey(date = new Date()) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

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
  const [earlyLeavePopup, setEarlyLeavePopup] = useState(null);

  const pollIntervalRef = useRef(null);
  const facePollRef = useRef(null);
  const clockRef = useRef(null);
  const earlyLeavePopupRef = useRef({ lastKey: null, timer: null });

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
    const today = getLocalDateKey();
    Promise.all([
      getAttendanceLogs({ date: today, limit: 20 }),
      getReportSummary('day'),
    ]).then(([logsRes, summaryRes]) => {
      const fetched = logsRes.data || [];
      setLogs(fetched);

      const totalEmp = (summaryRes && !summaryRes.error) ? (summaryRes.total_employees || 10) : 10;
      const todayLogs = fetched;

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

  useEffect(() => {
    const latestEarlyLeave = logs.find(log => log.status === 'EARLY_LEAVE');
    if (!latestEarlyLeave?.check_time) return;

    const checkTime = new Date(latestEarlyLeave.check_time);
    if (Number.isNaN(checkTime.getTime()) || Date.now() - checkTime.getTime() > 10000) return;

    const popupKey = latestEarlyLeave.attendance_id || `${latestEarlyLeave.employee_id}-${latestEarlyLeave.check_time}`;
    if (earlyLeavePopupRef.current.lastKey === popupKey) return;

    earlyLeavePopupRef.current.lastKey = popupKey;
    window.clearTimeout(earlyLeavePopupRef.current.timer);
    setEarlyLeavePopup(latestEarlyLeave);
    earlyLeavePopupRef.current.timer = window.setTimeout(() => {
      setEarlyLeavePopup(null);
    }, 8000);
  }, [logs]);

  useEffect(() => {
    return () => window.clearTimeout(earlyLeavePopupRef.current.timer);
  }, []);

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

  const timeStr = now.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const dateStr = now.toLocaleDateString('vi-VN', { weekday: 'long', day: '2-digit', month: '2-digit', year: 'numeric' });

  return (
    <div className="animate-in" style={{ maxWidth: 1280, margin: '0 auto', paddingBottom: '3rem' }}>
      {earlyLeavePopup && (
        <div style={{
          position: 'fixed', top: 24, right: 24, zIndex: 9999,
          width: 'min(420px, calc(100vw - 32px))',
          background: '#fff7ed', color: '#9a3412',
          border: '1px solid #fdba74', borderLeft: '5px solid #f97316',
          borderRadius: '12px', padding: '1rem 1.1rem',
          boxShadow: '0 18px 45px rgba(15, 23, 42, 0.22)',
          display: 'flex', gap: '0.85rem', alignItems: 'flex-start',
          animation: 'fadeIn 0.2s ease'
        }}>
          <div style={{
            width: 36, height: 36, borderRadius: '50%',
            background: '#ffedd5', display: 'flex', alignItems: 'center',
            justifyContent: 'center', flexShrink: 0
          }}>
            <TrendingDown size={20} color="#ea580c" />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: '0.78rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
              Canh bao check-out ve som
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: 800, color: '#7c2d12', lineHeight: 1.3 }}>
              {earlyLeavePopup.full_name || earlyLeavePopup.employee_id || 'Nhan vien'} dang ra ve truoc gio quy dinh
            </div>
            <div style={{ marginTop: '0.35rem', fontSize: '0.78rem', fontWeight: 600, lineHeight: 1.45 }}>
              Gio tan lam: <strong>{policy.work_end_time}</strong> | Check-out: <strong>{formatTime(earlyLeavePopup.check_time)}</strong>
            </div>
          </div>
          <button
            type="button"
            aria-label="Dong canh bao ve som"
            onClick={() => {
              window.clearTimeout(earlyLeavePopupRef.current.timer);
              setEarlyLeavePopup(null);
            }}
            style={{
              border: 0, background: 'transparent', color: '#9a3412',
              fontSize: '1.2rem', lineHeight: 1, cursor: 'pointer',
              padding: 0, width: 24, height: 24, fontWeight: 800
            }}
          >
            x
          </button>
        </div>
      )}

      {/* ── Header ── */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: '1.5rem', borderBottom: '1px solid var(--br)', paddingBottom: '1.25rem',
        flexWrap: 'wrap', gap: '1rem'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.25rem' }}>
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: isStreaming ? '#10b981' : '#64748b',
              boxShadow: isStreaming ? '0 0 10px #10b981' : 'none',
              animation: isStreaming ? 'pulse 1.5s infinite' : 'none'
            }} />
            <span style={{ fontSize: '0.75rem', fontWeight: 800, color: isStreaming ? '#10b981' : '#64748b', letterSpacing: '0.1em' }}>
              {isStreaming ? 'HỆ THỐNG ĐANG QUÉT' : 'HỆ THỐNG NGOẠI TUYẾN'}
            </span>
          </div>
          <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--tx-1)', letterSpacing: '-0.02em', margin: 0 }}>
            Biometric Attendance Center
          </h1>
          <p style={{ color: 'var(--tx-3)', fontSize: '0.85rem', marginTop: '0.15rem' }}>
            Nhận dạng khuôn mặt 3D · Anti-Spoofing (Liveness) · Đồng bộ Supabase
          </p>
        </div>

        {/* Action + Clock Panel */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {/* Policy indicator */}
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'flex-end',
            background: 'var(--bg-subtle)', border: '1px solid var(--br)',
            padding: '0.5rem 0.85rem', borderRadius: '10px',
            fontSize: '0.78rem', color: 'var(--tx-2)', fontWeight: 600
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--ac)', fontWeight: 700, marginBottom: '0.15rem' }}>
              <ShieldCheck size={13} />
              Cấu hình Ca làm việc
            </div>
            <div>
              Ca sáng: <strong>{policy.work_start_time}</strong> (+{policy.allow_late_minutes}m) | Ca chiều: <strong>{policy.work_end_time}</strong>
            </div>
          </div>

          {/* Clock widget */}
          <div style={{
            background: '#1e293b', color: '#f8fafc',
            padding: '0.45rem 1rem', borderRadius: '10px',
            border: '1px solid #334155', display: 'flex', flexDirection: 'column', alignItems: 'center'
          }}>
            <div style={{ fontFamily: 'monospace', fontSize: '1.35rem', fontWeight: 800, letterSpacing: '0.05em', lineHeight: 1.1 }}>
              {timeStr}
            </div>
            <div style={{ fontSize: '0.62rem', color: '#94a3b8', marginTop: '0.1rem', textTransform: 'uppercase', fontWeight: 700 }}>
              {dateStr.split(',')[0] || dateStr}
            </div>
          </div>
        </div>
      </div>

      {/* ── Stats row ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        {[
          { label: 'Đã vào', value: stats.present, sub: `/ ${stats.total} nhân viên`, color: '#10b981', Icon: UserCheck, bgGlow: 'rgba(16, 185, 129, 0.08)' },
          { label: 'Vào muộn', value: stats.late, sub: 'lượt đi muộn', color: '#f59e0b', Icon: Clock, bgGlow: 'rgba(245, 158, 11, 0.08)' },
          { label: 'Về sớm', value: stats.earlyLeave, sub: 'lượt về sớm', color: '#f97316', Icon: TrendingDown, bgGlow: 'rgba(249, 115, 22, 0.08)' },
          { label: 'Vắng mặt', value: stats.absent, sub: 'chưa check-in', color: '#ef4444', Icon: UserX, bgGlow: 'rgba(239, 68, 68, 0.08)' },
          { label: 'Chuyên cần', value: `${stats.rate}%`, sub: 'tỷ lệ hôm nay', color: '#3b82f6', Icon: Percent, bgGlow: 'rgba(59, 130, 246, 0.08)' },
        ].map(({ label, value, sub, color, Icon, bgGlow }) => (
          <div key={label} className="stat-glass-card" style={{ display: 'flex', alignItems: 'center', gap: '0.9rem' }}>
            <div className="stat-glass-icon" style={{ background: bgGlow }}>
              <Icon size={20} color={color} />
            </div>
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--tx-3)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--tx-1)', lineHeight: 1.25, marginTop: '0.1rem' }}>
                {value} <span style={{ fontSize: '0.75rem', fontWeight: 500, color: 'var(--tx-3)' }}>{sub}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* ── Main 2-col ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: '1.5rem', alignItems: 'start' }}>

        {/* Left: Camera Viewport & Stream Control */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--tx-2)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#3b82f6' }} />
              Realtime Video Capture
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <select
                value={selectedCameraId}
                onChange={e => { setSelectedCameraId(e.target.value); setStreamError(false); }}
                style={{
                  fontSize: '0.75rem', padding: '0.25rem 0.6rem',
                  borderRadius: '6px', border: '1px solid var(--br)',
                  background: 'var(--bg-surface)', color: 'var(--tx-1)', fontWeight: 600, outline: 'none'
                }}
              >
                <option value="system">Hệ thống (Mặc định)</option>
                <option value="0">Webcam 0 (Chính)</option>
                <option value="1">Webcam 1 (Phụ)</option>
                <option value="2">Webcam 2</option>
              </select>
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: '0.25rem',
                background: isStreaming ? '#ecfdf5' : '#f1f5f9',
                color: isStreaming ? '#059669' : '#64748b',
                padding: '0.25rem 0.65rem', borderRadius: '50px', fontSize: '0.7rem', fontWeight: 800
              }}>
                {isStreaming ? <Wifi size={11} className="pulse" /> : <WifiOff size={11} />}
                {isStreaming ? 'STREAMING' : 'OFFLINE'}
              </span>
            </div>
          </div>

          {/* Camera viewport */}
          <div style={{
            background: '#070a13', borderRadius: '12px', minHeight: 420,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            overflow: 'hidden', position: 'relative',
            border: '1.5px solid #1e293b', boxShadow: '0 10px 30px rgba(0, 0, 0, 0.25), inset 0 0 40px rgba(0, 0, 0, 0.8)'
          }}>
            {isStreaming && streamUrl ? (
              <>
                <img
                  src={streamUrl}
                  style={{ width: '100%', height: '100%', minHeight: 420, objectFit: 'contain', zIndex: 1 }}
                  alt="Live Stream Feed"
                  onError={() => { setIsStreaming(false); setStreamError(true); }}
                />

                {/* HIGH-TECH HUD OVERLAYS */}
                <div className="hud-corner tl" />
                <div className="hud-corner tr" />
                <div className="hud-corner bl" />
                <div className="hud-corner br" />

                {/* Laser animation */}
                <div className="scan-laser-line" />

                {/* Info Hud Box (Bottom Left) */}
                <div style={{
                  position: 'absolute', bottom: '1rem', left: '1rem',
                  fontFamily: 'monospace', fontSize: '0.68rem', zIndex: 10,
                  background: 'rgba(9,13,22,0.85)', padding: '0.6rem 0.85rem',
                  borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)', color: '#94a3b8',
                  backdropFilter: 'blur(4px)'
                }}>
                  <div style={{ color: '#22c55e', fontWeight: 800, marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', display: 'inline-block' }} />
                    ACQUISITION ACTIVE
                  </div>
                  <div>MODEL: InsightFace ResNet50</div>
                  <div>DEVICE ID: {selectedCameraId.toUpperCase()}</div>
                  <div>FPS: ~30.0 | THRESHOLD: 45%</div>
                </div>

                {/* Info Hud Box (Top Left) */}
                <div style={{
                  position: 'absolute', top: '1rem', left: '1rem', zIndex: 10,
                  background: 'rgba(9,13,22,0.85)', padding: '0.35rem 0.65rem',
                  borderRadius: '6px', border: '1px solid rgba(255,255,255,0.08)',
                  display: 'flex', alignItems: 'center', gap: '0.35rem', backdropFilter: 'blur(4px)'
                }}>
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#ef4444', display: 'inline-block', animation: 'pulse 1s infinite' }} />
                  <span style={{ color: '#f8fafc', fontSize: '0.65rem', fontWeight: 800, fontFamily: 'monospace', letterSpacing: '0.05em' }}>LIVE FEED</span>
                </div>

                {/* Last recognition flash */}
                {isRecent && latestLog && (
                  <div style={{
                    position: 'absolute', top: '1rem', right: '1rem', zIndex: 10,
                    background: 'rgba(15, 23, 42, 0.9)', border: `1px solid ${STATUS_CONFIG[latestLog.status]?.border || '#334155'}`,
                    padding: '0.5rem 0.85rem', borderRadius: '8px', fontSize: '0.75rem',
                    fontFamily: 'monospace', color: STATUS_CONFIG[latestLog.status]?.color || '#fff',
                    boxShadow: `0 4px 15px ${STATUS_CONFIG[latestLog.status]?.color}33`,
                    backdropFilter: 'blur(4px)'
                  }}>
                    MATCH: <strong>{latestLog.full_name}</strong> ({Math.round((latestLog.similarity || 0.8) * 100)}%)
                  </div>
                )}
              </>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flexGrow: 1, padding: '3rem', color: '#475569', textAlign: 'center' }}>
                {/* Radar visualization inside offline screen */}
                <div className="radar-container" style={{ opacity: 0.3, marginBottom: '1rem', width: 100, height: 100 }}>
                  <div className="radar-sweep" />
                  <div className="radar-grid-circle c1" />
                  <div className="radar-grid-circle c2" />
                </div>
                <h3 style={{ margin: 0, fontWeight: 700, color: '#94a3b8', fontSize: '0.95rem' }}>Camera Standby</h3>
                <p style={{ fontSize: '0.78rem', color: '#475569', marginTop: '0.25rem', maxWidth: '280px' }}>
                  Hệ thống nhận diện đang tạm dừng. Bấm nút phía dưới để kích hoạt camera.
                </p>
                {streamError && (
                  <div style={{ color: '#ef4444', fontSize: '0.75rem', fontWeight: 600, marginTop: '0.75rem', background: 'rgba(239, 68, 68, 0.05)', padding: '0.4rem 0.8rem', borderRadius: '6px', border: '1px solid rgba(239, 68, 68, 0.15)' }}>
                    Error: Camera offline hoặc bị chiếm bởi tiến trình khác.
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Action button below view */}
          <div style={{ marginTop: '0.85rem' }}>
            <button
              className={`btn ${isStreaming ? 'btn-danger' : 'btn-primary'}`}
              style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', height: '42px', borderRadius: '10px' }}
              onClick={() => {
                setStreamError(false);
                if (isStreaming) {
                  setIsStreaming(false);
                  setCurrentFace(null);
                  setCurrentFaceTs(null);
                } else {
                  setIsStreaming(true);
                }
              }}
            >
              {isStreaming ? (
                <>
                  <Square size={14} fill="currentColor" /> DỪNG QUÉT CAMERA
                </>
              ) : (
                <>
                  <Play size={14} fill="currentColor" /> BẮT ĐẦU QUÉT CAMERA
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right: Biometric Matching Panel */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', minHeight: 495, padding: '1.25rem' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--tx-2)', textTransform: 'uppercase', letterSpacing: '0.06em', borderBottom: '1px solid var(--br)', paddingBottom: '0.65rem', marginBottom: '1rem' }}>
            👤 Biometric Recognition Panel
          </div>

          {currentFace ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.85rem', animation: 'fadeIn 0.25s ease' }}>

              {/* Portrait ring */}
              <div style={{
                width: 130, height: 130, borderRadius: '50%', overflow: 'hidden',
                border: `4px solid ${ringColor}`,
                boxShadow: `0 0 25px ${ringColor}44`,
                background: 'var(--bg-subtle)', flexShrink: 0,
                position: 'relative'
              }}>
                <Portrait empId={faceEmpId} name={currentFace.full_name} size={130} backend={BACKEND} />
                <div style={{
                  position: 'absolute', bottom: 4, right: 4, width: 22, height: 22,
                  borderRadius: '50%', background: ringColor, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  border: '2px solid #fff', boxShadow: '0 2px 6px rgba(0,0,0,0.15)'
                }}>
                  <StatusIcon status={currentFace.status} size={10} color="#fff" />
                </div>
              </div>

              {/* ID Badge details */}
              <div style={{ textAlign: 'center' }}>
                <h2 style={{ fontSize: '1.35rem', fontWeight: 800, margin: '0 0 0.2rem', color: 'var(--tx-1)', letterSpacing: '-0.01em' }}>
                  {currentFace.full_name || 'Đang phân tích...'}
                </h2>
                {faceEmpId ? (
                  <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--tx-3)', fontWeight: 600 }}>
                    MÃ SỐ: <span style={{ color: 'var(--ac)' }}>{faceEmpId}</span>
                    {(logs.find(l => l.employee_id === faceEmpId)?.department) &&
                      <> &nbsp;·&nbsp; PHÒNG: <span style={{ color: 'var(--tx-2)' }}>{logs.find(l => l.employee_id === faceEmpId).department.toUpperCase()}</span></>
                    }
                  </p>
                ) : (
                  <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--tx-3)' }}>Khách truy cập / Chưa đăng ký</p>
                )}
              </div>

              {/* Status Tag */}
              <div style={{ transform: 'scale(1.05)', margin: '0.2rem 0' }}>
                <StatusBadge status={currentFace.status} size="lg" />
              </div>

              {/* Dynamic Notification notes */}
              {currentFace.status === 'LATE' && (
                <div style={{ background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '8px', padding: '0.5rem 0.75rem', fontSize: '0.75rem', color: '#b45309', fontWeight: 600, textAlign: 'center', width: '100%' }}>
                  ⏰ Đi muộn so với giờ quy định ({policy.work_start_time})
                </div>
              )}
              {currentFace.status === 'EARLY_LEAVE' && (
                <div style={{ background: '#fff7ed', border: '1px solid #fed7aa', borderRadius: '8px', padding: '0.5rem 0.75rem', fontSize: '0.75rem', color: '#c2410c', fontWeight: 600, textAlign: 'center', width: '100%' }}>
                  🏃 Về trước giờ tan làm theo quy định ({policy.work_end_time})
                </div>
              )}
              {currentFace.status === 'UNKNOWN' && (
                <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '8px', padding: '0.5rem 0.75rem', fontSize: '0.75rem', color: '#b91c1c', fontWeight: 600, textAlign: 'center', width: '100%' }}>
                  ❌ Gương mặt không khớp dữ liệu nhân viên
                </div>
              )}
              {currentFace.status === 'SPOOFING' && (
                <div style={{ background: '#fef2f2', border: '1px solid #f87171', borderRadius: '8px', padding: '0.5rem 0.75rem', fontSize: '0.75rem', color: '#b91c1c', fontWeight: 700, textAlign: 'center', width: '100%' }}>
                  🚨 CẢNH BÁO: Phát hiện ảnh chụp/video giả mạo!
                </div>
              )}
              {currentFace.status === 'REJECTED_CHECK_IN' && (
                <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '8px', padding: '0.5rem 0.75rem', fontSize: '0.75rem', color: '#b91c1c', fontWeight: 600, textAlign: 'center', width: '100%' }}>
                  ❌ Từ chối check-in: Đã quá thời gian quy định (sau 12:00)
                </div>
              )}
              {currentFace.status === 'REJECTED_CHECK_OUT' && (
                <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '8px', padding: '0.5rem 0.75rem', fontSize: '0.75rem', color: '#b91c1c', fontWeight: 600, textAlign: 'center', width: '100%' }}>
                  ❌ Từ chối check-out: Chưa đến thời gian quy định (trước 12:00)
                </div>
              )}
              {currentFace.status === 'COMPLETED' && (
                <div style={{ background: '#ecfdf5', border: '1px solid #6ee7b7', borderRadius: '8px', padding: '0.5rem 0.75rem', fontSize: '0.75rem', color: '#047857', fontWeight: 600, textAlign: 'center', width: '100%' }}>
                  ✅ Đã hoàn tất cả check-in và check-out hôm nay!
                </div>
              )}

              {/* Stats & Match Values */}
              <div style={{
                width: '100%', background: 'var(--bg-subtle)', borderRadius: '10px',
                padding: '0.85rem', border: '1px solid var(--br)', display: 'flex', flexDirection: 'column', gap: '0.75rem'
              }}>
                <ConfidenceBar value={currentFace.similarity} />
                <LivenessBar value={currentFace.liveness_score} />

                {currentFace.check_time && (
                  <div style={{ borderTop: '1px solid var(--br)', paddingTop: '0.65rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <span style={{ fontSize: '0.65rem', color: 'var(--tx-3)', fontWeight: 700, display: 'block' }}>THỜI GIAN ĐIỂM DANH</span>
                      <span style={{ fontFamily: 'monospace', fontSize: '0.88rem', fontWeight: 800, color: 'var(--tx-1)' }}>{formatTime(currentFace.check_time)}</span>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontSize: '0.65rem', color: 'var(--tx-3)', fontWeight: 700, display: 'block' }}>NGÀY GHI NHẬN</span>
                      <span style={{ fontSize: '0.78rem', color: 'var(--tx-2)', fontWeight: 600 }}>{formatDate(currentFace.check_time)}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flexGrow: 1, padding: '2rem 1rem' }}>
              {/* Radar Scanner Animation for Idle state */}
              <div className="radar-container">
                <div className="radar-sweep" />
                <div className="radar-grid-circle c1" />
                <div className="radar-grid-circle c2" />
                <div className="radar-grid-circle c3" />
                <div className="radar-dot" style={{ top: '30%', left: '70%' }} />
                <div className="radar-dot" style={{ top: '65%', left: '20%', animationDelay: '0.7s' }} />
              </div>
              <p style={{ margin: '1rem 0 0.25rem', fontWeight: 700, fontSize: '0.85rem', color: 'var(--tx-2)', letterSpacing: '0.05em' }}>
                ĐANG QUÉT KHUÔN MẶT...
              </p>
              <p style={{ margin: 0, fontSize: '0.75rem', textAlign: 'center', color: 'var(--tx-3)', maxWidth: '240px', lineHeight: 1.4 }}>
                Vui lòng đứng thẳng, chính diện trước camera để ghi nhận chấm công tự động.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── Log table ── */}
      <div className="card" style={{ marginTop: '1.5rem', padding: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--br)', paddingBottom: '0.75rem', marginBottom: '1rem' }}>
          <span style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--tx-2)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Users size={14} color="var(--ac)" />
            Nhật ký điểm danh hôm nay
          </span>
          <button
            className="btn btn-sm btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', borderRadius: '6px' }}
            onClick={loadData}
          >
            <RefreshCw size={11} /> Làm mới log
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
                  <th>Mã số</th>
                  <th>Họ tên</th>
                  <th>Phòng ban</th>
                  <th style={{ textAlign: 'center' }}>Hình thức</th>
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
                    <tr key={i} style={i === 0 && isRecent ? { background: `${cfg.bg}aa` } : {}}>
                      {/* Portrait */}
                      <td style={{ textAlign: 'center', padding: '0.45rem' }}>
                        <div style={{
                          width: 34, height: 34, borderRadius: '50%',
                          overflow: 'hidden', border: `2px solid ${cfg.border}`,
                          background: 'var(--bg-subtle)', margin: '0 auto',
                          boxShadow: i === 0 && isRecent ? `0 0 10px ${cfg.border}` : 'none'
                        }}>
                          <Portrait empId={log.employee_id} name={log.full_name} size={34} backend={BACKEND} />
                        </div>
                      </td>
                      {/* Time */}
                      <td style={{ fontFamily: 'monospace', fontSize: '0.85rem', fontWeight: 800, color: 'var(--tx-1)' }}>
                        {formatTime(log.check_time)}
                      </td>
                      {/* Date */}
                      <td style={{ fontSize: '0.78rem', color: 'var(--tx-3)' }}>
                        {formatDate(log.check_time)}
                      </td>
                      {/* Employee ID */}
                      <td>
                        <strong style={{ color: 'var(--ac)', fontSize: '0.82rem', fontFamily: 'monospace' }}>{log.employee_id}</strong>
                      </td>
                      {/* Name */}
                      <td style={{ fontWeight: 650, fontSize: '0.88rem', color: 'var(--tx-1)' }}>{log.full_name || '—'}</td>
                      {/* Department */}
                      <td style={{ fontSize: '0.8rem', color: 'var(--tx-2)', fontWeight: 500 }}>{log.department || '—'}</td>
                      {/* Check-in / out */}
                      <td style={{ textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-flex', alignItems: 'center', gap: '0.25rem',
                          fontSize: '0.7rem', fontWeight: 800,
                          color: isCheckOut ? '#3b82f6' : '#10b981'
                        }}>
                          {isCheckOut ? <LogOut size={11} /> : <LogIn size={11} />}
                          {isCheckOut ? 'RA' : 'VÀO'}
                        </span>
                      </td>
                      {/* Similarity */}
                      <td style={{ textAlign: 'center' }}>
                        {sim !== null ? (
                          <span style={{
                            fontFamily: 'monospace', fontWeight: 800, fontSize: '0.82rem',
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
                            fontFamily: 'monospace', fontWeight: 800, fontSize: '0.82rem',
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
          <div className="empty-state" style={{ padding: '3rem' }}>
            <Users size={36} style={{ opacity: 0.25, marginBottom: '0.75rem', color: 'var(--tx-3)' }} />
            <p style={{ margin: 0, fontWeight: 500 }}>Chưa ghi nhận lượt chấm công nào trong hôm nay</p>
          </div>
        )}
      </div>

    </div>
  );
}
