import { Camera } from 'lucide-react';

export default function FaceRegistration() {
  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Face Registration</h1>
        <p>Đăng ký khuôn mặt nhân viên</p>
      </div>

      <div className="grid-2">
        {/* Webcam Area */}
        <div className="card">
          <div className="card-header"><span className="card-title">🎥 Webcam Preview</span></div>
          <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-md)', height: 350, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem', border: '2px dashed var(--border-glow)' }}>
            <Camera size={48} style={{ opacity: 0.3 }} />
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Camera Preview</p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Chạy: <code>python -m src.attendance.register_employee</code></p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
            <button className="btn btn-primary" style={{ flex: 1 }}>▶️ Start Capture</button>
            <button className="btn btn-secondary" style={{ flex: 1 }}>⏹️ Stop</button>
            <button className="btn btn-secondary" style={{ flex: 1 }}>🧠 Build Embedding</button>
          </div>
        </div>

        {/* Info & Metrics */}
        <div>
          <div className="card" style={{ marginBottom: '1.25rem' }}>
            <div className="card-header"><span className="card-title">📋 Thông Tin NV</span></div>
            <div className="input-group"><label>Employee ID</label><input className="input" placeholder="NV001" /></div>
            <div className="input-group"><label>Họ tên</label><input className="input" placeholder="Nguyễn Văn A" /></div>
            <div className="input-group"><label>Phòng ban</label>
              <select className="input"><option>IT</option><option>HR</option><option>Finance</option><option>Marketing</option></select>
            </div>
            <div className="input-group"><label>Chức vụ</label><input className="input" placeholder="Developer" /></div>
          </div>

          <div className="card">
            <div className="card-header"><span className="card-title">📊 Chất Lượng</span></div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-info)' }}>0/100</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Ảnh đã thu</div>
              </div>
              <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-success)' }}>—</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Blur Score</div>
              </div>
              <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-warning)' }}>—</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Brightness</div>
              </div>
              <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-secondary)' }}>—</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Face Quality</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
