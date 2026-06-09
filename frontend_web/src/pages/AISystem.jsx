import { useEffect, useState } from 'react';
import { RefreshCw, Trash2, Database, Shield, Cpu, CheckCircle2, AlertCircle } from 'lucide-react';
import { listEmbeddings, deleteEmbedding, rebuildEmbeddings, getModelInfo } from '../api';

export default function AISystem() {
  const [tab, setTab] = useState('embeddings');
  const [embeddings, setEmbeddings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedDs, setSelectedDs] = useState('LFW');

  const loadData = () => {
    setLoading(true);
    Promise.all([listEmbeddings(), getModelInfo()]).then(([embeddingsRes, modelInfoRes]) => {
      setEmbeddings(embeddingsRes.data || []);
      if (modelInfoRes && !modelInfoRes.error) {
        setModelInfo(modelInfoRes);
      }
      setLoading(false);
    });
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleDelete = async (id) => {
    if (confirm(`Bạn có chắc chắn muốn xóa embedding của nhân viên ${id}? Hành động này sẽ xóa dữ liệu vector nhận diện.`)) {
      setProcessing(true);
      const res = await deleteEmbedding(id);
      setProcessing(false);
      if (res && !res.error) {
        setMessage({ type: 'success', text: `Đã xóa thành công embedding của nhân viên ${id}.` });
        loadData();
      } else {
        setMessage({ type: 'error', text: `Lỗi xóa: ${res?.error || 'Không thể xóa'}` });
      }
    }
  };

  const handleRebuildAll = async () => {
    setProcessing(true);
    setMessage({ type: '', text: '' });
    const res = await rebuildEmbeddings();
    setProcessing(false);
    if (res && !res.error) {
      setMessage({ type: 'success', text: 'Đã trích xuất và tạo mới toàn bộ dữ liệu Vector Face Embeddings thành công!' });
      loadData();
    } else {
      setMessage({ type: 'error', text: `Lỗi tạo lại: ${res?.error || 'Không thể chạy quy trình trích xuất'}` });
    }
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Quản trị AI & Sinh trắc học</h1>
        <p>Quản lý cơ sở dữ liệu Face Vector, chạy đánh giá độ chính xác mô hình và giám sát chống giả mạo</p>
      </div>

      <div className="tabs">
        <button className={`tab ${tab === 'embeddings' ? 'active' : ''}`} onClick={() => setTab('embeddings')}>📦 Dữ liệu Vector AI</button>
        <button className={`tab ${tab === 'evaluation' ? 'active' : ''}`} onClick={() => setTab('evaluation')}>📊 Đánh giá Mô hình</button>
        <button className={`tab ${tab === 'spoofing' ? 'active' : ''}`} onClick={() => setTab('spoofing')}>🛡️ Chống giả mạo (Anti-Spoofing)</button>
      </div>

      {message.text && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          background: message.type === 'success' ? '#ecfdf5' : '#fef2f2',
          border: `1px solid ${message.type === 'success' ? '#34d399' : '#f87171'}`,
          color: message.type === 'success' ? '#065f46' : '#991b1b',
          padding: '0.85rem 1rem',
          borderRadius: 'var(--radius-md)',
          marginBottom: '1.25rem',
          fontSize: '0.85rem',
          fontWeight: 500
        }}>
          {message.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
          <span>{message.text}</span>
        </div>
      )}

      {tab === 'embeddings' && (
        <div>
          <div className="card">
            <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Database size={16} style={{ color: 'var(--accent-primary)' }} />
                Cơ sở dữ liệu Vector Face Embeddings ({embeddings.length})
              </span>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="btn btn-sm btn-secondary" onClick={loadData} disabled={loading || processing}>
                  <RefreshCw size={12} className={loading ? 'spin' : ''} style={{ animation: loading ? 'spin 2s linear infinite' : 'none' }} /> Làm mới
                </button>
                <button className="btn btn-sm btn-primary" onClick={handleRebuildAll} disabled={processing || loading} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <RefreshCw size={12} className={processing ? 'spin' : ''} style={{ animation: processing ? 'spin 2s linear infinite' : 'none' }} />
                  {processing ? 'Đang trích xuất...' : 'Trích xuất lại toàn bộ'}
                </button>
              </div>
            </div>

            {loading ? (
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                <RefreshCw size={24} className="spin" style={{ animation: 'spin 2s linear infinite', marginRight: '0.5rem' }} />
                <span>Đang nạp danh sách vector...</span>
              </div>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Mã nhân viên</th>
                    <th>Họ và tên</th>
                    <th>Số lượng ảnh mẫu</th>
                    <th>Độ dài Vector</th>
                    <th>Cập nhật lần cuối</th>
                    <th style={{ textAlign: 'right' }}>Thao tác</th>
                  </tr>
                </thead>
                <tbody>
                  {embeddings.map(emb => (
                    <tr key={emb.employee_id}>
                      <td><strong style={{ color: 'var(--accent-primary)' }}>{emb.employee_id}</strong></td>
                      <td style={{ fontWeight: 600 }}>{emb.full_name || '—'}</td>
                      <td><span className="badge badge-info">{emb.image_count || 0} ảnh chân dung</span></td>
                      <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>512-dim</td>
                      <td style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        {emb.updated_at ? new Date(emb.updated_at).toLocaleString('vi-VN') : '—'}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <button
                          className="btn-icon"
                          style={{ color: 'var(--accent-danger)' }}
                          onClick={() => handleDelete(emb.employee_id)}
                          title="Xóa Vector"
                          disabled={processing}
                        >
                          <Trash2 size={14} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {!loading && embeddings.length === 0 && (
              <div className="empty-state" style={{ padding: '3rem 2rem' }}>
                <Database size={36} style={{ opacity: 0.3, marginBottom: '0.5rem' }} />
                <p style={{ margin: 0 }}>Chưa có dữ liệu Vector Face Embeddings. Vui lòng đăng ký nhân viên mới.</p>
              </div>
            )}
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>
            * Face Embeddings là các vector số học 512 chiều trích xuất từ khuôn mặt thông qua mô hình học sâu ArcFace. Cơ chế đối sánh sẽ sử dụng khoảng cách Cosine trên các vector này để định danh nhân sự.
          </p>
        </div>
      )}

      {tab === 'evaluation' && (
        <div>
          <div className="card" style={{ marginBottom: '1.25rem' }}>
            <div className="card-header"><span className="card-title"><Cpu size={16} /> Thông số mô hình nhận diện đang nạp</span></div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-primary)' }}>{modelInfo?.model_name || 'arcface_vggface2_warmup.pth'}</div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Kiến trúc Backbone: ResNet50 | Vector đặc trưng: 512 dimensions | Huấn luyện trên tập: VGGFace2</p>
          </div>

          <h4 style={{ marginBottom: '0.75rem', fontWeight: 600 }}>📋 Đánh giá trên tập AFDB Masked Face (Nhận diện đeo khẩu trang)</h4>
          <div className="stats-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
            {[
              { label: 'Rank-1 Accuracy', value: modelInfo?.metrics?.rank1 || '—', color: 'blue' },
              { label: 'Rank-5 Accuracy', value: modelInfo?.metrics?.rank5 || '—', color: 'green' },
              { label: 'Equal Error Rate (EER)', value: modelInfo?.metrics?.eer || '—', color: 'red' },
              { label: 'Ngưỡng tối ưu EER', value: modelInfo?.metrics?.threshold || '—', color: 'yellow' },
            ].map(s => (
              <div className={`stat-card ${s.color}`} key={s.label} style={{ padding: '1.25rem' }}>
                <div className="stat-value" style={{ fontSize: '1.8rem', fontWeight: 800 }}>{s.value}</div>
                <div className="stat-label" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{s.label}</div>
              </div>
            ))}
          </div>

          {modelInfo?.benchmarks && modelInfo.benchmarks.length > 0 && (
            <div style={{ marginTop: '2rem', marginBottom: '2rem' }}>
              <h4 style={{ marginBottom: '0.75rem', fontWeight: 600 }}>📊 Kết quả kiểm thử trên các tập dữ liệu tiêu chuẩn (Benchmarks)</h4>
              <table className="data-table" style={{ width: '100%', marginBottom: '1.5rem' }}>
                <thead>
                  <tr>
                    <th>Dataset</th>
                    <th>Số cặp (Pairs)</th>
                    <th>Độ chính xác (Accuracy)</th>
                    <th>AUC</th>
                    <th>EER</th>
                    <th>Ngưỡng tối ưu</th>
                    <th>FAR (Nhận nhầm)</th>
                    <th>FRR (Từ chối sai)</th>
                  </tr>
                </thead>
                <tbody>
                  {modelInfo.benchmarks.map(b => (
                    <tr key={b.dataset}>
                      <td><strong>{b.dataset}</strong></td>
                      <td>{b.pairs}</td>
                      <td style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>{b.accuracy}</td>
                      <td>{b.auc}</td>
                      <td>{b.eer}</td>
                      <td>{b.threshold}</td>
                      <td>{b.far}</td>
                      <td>{b.frr}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h4 style={{ marginBottom: '0.75rem', fontWeight: 600 }}>📈 Biểu đồ Đánh giá Chi tiết (ROC & Confusion Matrix)</h4>
              <div style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>Chọn tập dữ liệu:</span>
                <select 
                  value={selectedDs} 
                  onChange={(e) => setSelectedDs(e.target.value)}
                  style={{
                    padding: '0.4rem 1rem',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-color)',
                    background: '#fff',
                    outline: 'none',
                    fontSize: '0.85rem'
                  }}
                >
                  <option value="LFW">LFW</option>
                  <option value="CALFW">CALFW</option>
                  <option value="CPLFW">CPLFW</option>
                  <option value="AGEDB">AGEDB</option>
                </select>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
                  <h5 style={{ marginBottom: '0.5rem', fontWeight: 600 }}>Ma trận nhầm lẫn (Confusion Matrix)</h5>
                  <img 
                    src={`http://localhost:8000/outputs/evaluation/${selectedDs === 'AGEDB' ? 'agedb30_eval' : selectedDs.toLowerCase() + '_eval'}_confusion_matrix.png`} 
                    alt={`${selectedDs} Confusion Matrix`}
                    style={{ width: '100%', borderRadius: 'var(--radius-md)' }}
                    onError={(e) => { e.target.style.display = 'none'; }}
                  />
                </div>
                <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
                  <h5 style={{ marginBottom: '0.5rem', fontWeight: 600 }}>Đường cong ROC (ROC Curve)</h5>
                  <img 
                    src={`http://localhost:8000/outputs/evaluation/${selectedDs === 'AGEDB' ? 'agedb30_eval' : selectedDs.toLowerCase() + '_eval'}_roc_curve.png`} 
                    alt={`${selectedDs} ROC Curve`}
                    style={{ width: '100%', borderRadius: 'var(--radius-md)' }}
                    onError={(e) => { e.target.style.display = 'none'; }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {tab === 'spoofing' && (
        <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
          <Shield size={64} style={{ color: '#10b981', marginBottom: '1.25rem', opacity: 0.8 }} />
          <h3 style={{ marginBottom: '0.5rem', fontWeight: 700 }}>Phát Hiện Thực Thể Sống (Liveness Detection)</h3>
          <p style={{ color: 'var(--text-muted)', maxWidth: '500px', margin: '0 auto 1.5rem' }}>
            Hệ thống sử dụng mạng nơ-ron học sâu để phát hiện các cuộc tấn công giả mạo (spoofing attack) bằng cách trình diện ảnh chân dung in trên giấy, video tái phát trên điện thoại hoặc mặt nạ silicon.
          </p>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', background: '#ecfdf5', border: '1px solid #10b981', color: '#047857', padding: '0.5rem 1.25rem', borderRadius: '50px', fontSize: '0.88rem', fontWeight: 700 }}>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#10b981', animation: 'ping 1.5s infinite' }} />
            <span>MÔ-ĐUN CHỐNG GIẢ MẠO ĐANG HOẠT ĐỘNG (LIVENESS ON)</span>
          </div>
        </div>
      )}
    </div>
  );
}
