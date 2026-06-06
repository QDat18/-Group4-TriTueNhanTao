import { useEffect, useState } from 'react';
import { RefreshCw, Trash2, Database, Shield, Cpu } from 'lucide-react';
import { listEmbeddings, deleteEmbedding } from '../api';

export default function AISystem() {
  const [tab, setTab] = useState('embeddings');
  const [embeddings, setEmbeddings] = useState([]);

  useEffect(() => { listEmbeddings().then(r => setEmbeddings(r.data || [])); }, []);

  const handleDelete = async (id) => {
    if (confirm(`Xóa embedding ${id}?`)) {
      await deleteEmbedding(id);
      listEmbeddings().then(r => setEmbeddings(r.data || []));
    }
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>AI System</h1>
        <p>Quản lý Embeddings, Đánh giá Mô hình, Anti-Spoofing</p>
      </div>

      <div className="tabs">
        <button className={`tab ${tab === 'embeddings' ? 'active' : ''}`} onClick={() => setTab('embeddings')}>📦 Embeddings</button>
        <button className={`tab ${tab === 'evaluation' ? 'active' : ''}`} onClick={() => setTab('evaluation')}>📊 Model Evaluation</button>
        <button className={`tab ${tab === 'spoofing' ? 'active' : ''}`} onClick={() => setTab('spoofing')}>🛡️ Anti-Spoofing</button>
      </div>

      {tab === 'embeddings' && (
        <div>
          <div className="card">
            <div className="card-header">
              <span className="card-title"><Database size={16} /> Embedding Database</span>
              <button className="btn btn-sm btn-primary"><RefreshCw size={14} /> Rebuild All</button>
            </div>
            <table className="data-table">
              <thead><tr><th>Employee</th><th>Tên</th><th>Số ảnh</th><th>Embedding Size</th><th>Cập nhật</th><th style={{ textAlign: 'right' }}>Actions</th></tr></thead>
              <tbody>
                {embeddings.map(emb => (
                  <tr key={emb.employee_id}>
                    <td><strong style={{ color: 'var(--accent-info)' }}>{emb.employee_id}</strong></td>
                    <td>{emb.full_name || '—'}</td>
                    <td><span className="badge badge-info">{emb.image_count || 0} ảnh</span></td>
                    <td style={{ fontFamily: 'monospace' }}>512</td>
                    <td style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{(emb.updated_at || '—').slice(0, 10)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button className="btn-icon" title="Rebuild"><RefreshCw size={14} /></button>{' '}
                      <button className="btn-icon" style={{ color: 'var(--accent-danger)' }} onClick={() => handleDelete(emb.employee_id)} title="Delete"><Trash2 size={14} /></button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {embeddings.length === 0 && <div className="empty-state"><p>Chưa có embeddings</p></div>}
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>Tổng: {embeddings.length} embeddings | Embedding size: 512</p>
        </div>
      )}

      {tab === 'evaluation' && (
        <div>
          <div className="card" style={{ marginBottom: '1.25rem' }}>
            <div className="card-header"><span className="card-title"><Cpu size={16} /> Current Model</span></div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-secondary)' }}>arcface_vggface2_warmup.pth</div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>ArcFace + ResNet | Embedding 512 | VGGFace2 + RMFRD</p>
          </div>

          <div className="stats-grid">
            {[
              { label: 'Accuracy', value: '—', color: 'blue' },
              { label: 'FAR', value: '—', color: 'red' },
              { label: 'FRR', value: '—', color: 'yellow' },
              { label: 'F1 Score', value: '—', color: 'green' },
            ].map(s => (
              <div className={`stat-card ${s.color}`} key={s.label}>
                <div className="stat-value">{s.value}</div>
                <div className="stat-label">{s.label}</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button className="btn btn-primary" style={{ flex: 1 }}>▶️ Run Evaluation</button>
            <button className="btn btn-secondary" style={{ flex: 1 }}>📥 Export Report</button>
          </div>
        </div>
      )}

      {tab === 'spoofing' && (
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <Shield size={64} style={{ color: 'var(--accent-primary)', marginBottom: '1rem', opacity: 0.5 }} />
          <h3 style={{ marginBottom: '0.5rem' }}>Liveness Detection</h3>
          <p style={{ color: 'var(--text-muted)' }}>Phát hiện giả mạo khuôn mặt (ảnh, video, mask)</p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '1rem' }}>Module đang phát triển...</p>
        </div>
      )}
    </div>
  );
}
