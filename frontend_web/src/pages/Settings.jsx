export default function Settings() {
  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Settings</h1>
        <p>Cấu hình hệ thống</p>
      </div>

      <div className="grid-2">
        <div>
          {/* Recognition Settings */}
          <div className="card" style={{ marginBottom: '1.25rem' }}>
            <div className="card-header"><span className="card-title">🎯 Recognition Settings</span></div>
            <div className="input-group">
              <label>Recognition Threshold</label>
              <input type="number" className="input" defaultValue={0.45} step={0.01} min={0} max={1} />
              <small style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>Ngưỡng similarity để nhận diện thành công</small>
            </div>
            <div className="input-group">
              <label>Cooldown (seconds)</label>
              <input type="number" className="input" defaultValue={60} min={10} max={3600} />
              <small style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>Thời gian chờ giữa 2 lần chấm công cùng NV</small>
            </div>
          </div>

          {/* Work Schedule */}
          <div className="card">
            <div className="card-header"><span className="card-title">🕐 Giờ Làm Việc</span></div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div className="input-group"><label>Giờ vào</label><input type="time" className="input" defaultValue="08:00" /></div>
              <div className="input-group"><label>Giờ ra</label><input type="time" className="input" defaultValue="17:30" /></div>
            </div>
            <div className="input-group">
              <label>Cho phép muộn (phút)</label>
              <input type="number" className="input" defaultValue={30} min={0} max={60} />
            </div>
          </div>
        </div>

        <div>
          {/* Database Info */}
          <div className="card" style={{ marginBottom: '1.25rem' }}>
            <div className="card-header"><span className="card-title">🗄️ Database</span></div>
            <div style={{ fontSize: '0.85rem', lineHeight: 2 }}>
              <div><span style={{ color: 'var(--text-muted)' }}>Provider:</span> <strong>Supabase (PostgreSQL)</strong></div>
              <div><span style={{ color: 'var(--text-muted)' }}>Host:</span> <strong>yxlvatmaiyhapjdguyco.supabase.co</strong></div>
              <div><span style={{ color: 'var(--text-muted)' }}>Tables:</span> <span className="badge badge-info">employees</span> <span className="badge badge-info">face_embeddings</span> <span className="badge badge-info">attendance_logs</span> <span className="badge badge-info">devices</span></div>
            </div>
          </div>

          {/* Model Info */}
          <div className="card">
            <div className="card-header"><span className="card-title">🤖 AI Model</span></div>
            <div style={{ fontSize: '0.85rem', lineHeight: 2 }}>
              <div><span style={{ color: 'var(--text-muted)' }}>Architecture:</span> <strong>ArcFace + ResNet</strong></div>
              <div><span style={{ color: 'var(--text-muted)' }}>Embedding Size:</span> <strong>512</strong></div>
              <div><span style={{ color: 'var(--text-muted)' }}>Checkpoint:</span> <strong style={{ color: 'var(--accent-secondary)' }}>arcface_vggface2_warmup.pth</strong></div>
              <div><span style={{ color: 'var(--text-muted)' }}>Training Data:</span> <strong>VGGFace2 + RMFRD</strong></div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <button className="btn btn-primary" style={{ width: '100%' }}>💾 Lưu Cấu Hình</button>
      </div>
    </div>
  );
}
