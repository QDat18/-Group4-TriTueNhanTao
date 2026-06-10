import { useState, useEffect } from 'react';
import { Save, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';
import { getSettings, updateSettings } from '../api';

export default function Settings() {
  const [form, setForm] = useState({
    work_start_time: '08:00',
    work_end_time: '17:30',
    allow_late_minutes: 30,
    allow_early_minutes: 15,
    cooldown_seconds: 43200,
    recognition_threshold: 0.45,
    camera_source_type: 'webcam',
    camera_webcam_index: 0,
    camera_ip_url: ''
  });
  
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  useEffect(() => {
    getSettings().then(res => {
      if (res && !res.error) {
        setForm({
          work_start_time: res.work_start_time || '08:00',
          work_end_time: res.work_end_time || '17:30',
          allow_late_minutes: res.allow_late_minutes !== undefined ? res.allow_late_minutes : 30,
          allow_early_minutes: res.allow_early_minutes !== undefined ? res.allow_early_minutes : 15,
          cooldown_seconds: res.cooldown_seconds !== undefined ? res.cooldown_seconds : 43200,
          recognition_threshold: res.recognition_threshold !== undefined ? res.recognition_threshold : 0.45,
          camera_source_type: res.camera_source_type || 'webcam',
          camera_webcam_index: res.camera_webcam_index !== undefined ? res.camera_webcam_index : 0,
          camera_ip_url: res.camera_ip_url || ''
        });
      }
      setLoading(false);
    });
  }, []);

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setMessage({ type: '', text: '' });
    
    // Parse values safely for the backend payload
    const payload = {
      work_start_time: form.work_start_time,
      work_end_time: form.work_end_time,
      allow_late_minutes: parseInt(form.allow_late_minutes) || 0,
      allow_early_minutes: parseInt(form.allow_early_minutes) || 0,
      cooldown_seconds: parseInt(form.cooldown_seconds) || 0,
      recognition_threshold: parseFloat(form.recognition_threshold) || 0.45,
      camera_source_type: form.camera_source_type,
      camera_webcam_index: parseInt(form.camera_webcam_index) || 0,
      camera_ip_url: form.camera_ip_url
    };
    
    const res = await updateSettings(payload);
    setSaving(false);
    
    if (res && !res.error) {
      setMessage({ type: 'success', text: 'Cấu hình hệ thống đã được cập nhật thành công!' });
      setTimeout(() => setMessage({ type: '', text: '' }), 5000);
    } else {
      setMessage({ type: 'error', text: `Lỗi: ${res?.error || 'Không thể kết nối API'}` });
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh', color: 'var(--text-muted)' }}>
        <RefreshCw className="spin" size={32} style={{ animation: 'spin 2s linear infinite' }} />
        <span style={{ marginLeft: '1rem', fontWeight: 500 }}>Đang tải cấu hình hệ thống...</span>
      </div>
    );
  }

  return (
    <div className="animate-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <div className="page-header">
        <h1>Cấu hình hệ thống</h1>
        <p>Tùy chỉnh thông số nhận diện khuôn mặt, thời gian làm việc và chính sách đi muộn</p>
      </div>

      {message.text && (
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '0.75rem', 
          background: message.type === 'success' ? '#ecfdf5' : '#fef2f2', 
          border: `1px solid ${message.type === 'success' ? '#6ee7b7' : '#fca5a5'}`, 
          color: message.type === 'success' ? '#047857' : '#b91c1c', 
          padding: '1rem', 
          borderRadius: 'var(--radius-md)', 
          marginBottom: '1.5rem',
          fontSize: '0.88rem',
          fontWeight: 500
        }}>
          {message.type === 'success' ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
          <span>{message.text}</span>
        </div>
      )}

      <form onSubmit={handleSave}>
        <div className="grid-2" style={{ gap: '1.5rem' }}>
          <div>
            {/* Recognition Settings */}
            <div className="card" style={{ marginBottom: '1.25rem' }}>
              <div className="card-header"><span className="card-title">🎯 Cấu hình nhận diện khuôn mặt</span></div>
              
              <div className="input-group">
                <label>Ngưỡng similarity nhận diện</label>
                <input 
                  type="number" 
                  className="input" 
                  value={form.recognition_threshold} 
                  onChange={e => setForm({ ...form, recognition_threshold: e.target.value })}
                  step={0.01} 
                  min={0.1} 
                  max={0.99} 
                  required
                />
                <small style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                  Độ tương đồng cosine tối thiểu để xác thực danh tính (khuyến nghị: 0.40 - 0.50)
                </small>
              </div>

              <div className="input-group">
                <label>Thời gian chờ Cooldown (giây)</label>
                <input 
                  type="number" 
                  className="input" 
                  value={form.cooldown_seconds} 
                  onChange={e => setForm({ ...form, cooldown_seconds: e.target.value })}
                  min={10} 
                  max={86400} 
                  required
                />
                <small style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                  Giãn cách tối thiểu giữa 2 lần nhận diện của cùng một người để tránh trùng lặp log
                </small>
              </div>
            </div>

            {/* Work Schedule */}
            <div className="card">
              <div className="card-header"><span className="card-title">🕐 Giờ làm việc & Đi muộn</span></div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="input-group">
                  <label>Giờ bắt đầu làm việc</label>
                  <input 
                    type="time" 
                    className="input" 
                    value={form.work_start_time} 
                    onChange={e => setForm({ ...form, work_start_time: e.target.value })}
                    required 
                  />
                </div>
                <div className="input-group">
                  <label>Phút ân hạn đi trễ</label>
                  <input 
                    type="number" 
                    className="input" 
                    value={form.allow_late_minutes} 
                    onChange={e => setForm({ ...form, allow_late_minutes: e.target.value })}
                    min={0} 
                    max={180} 
                    required 
                  />
                </div>
                <div className="input-group">
                  <label>Gio tan lam</label>
                  <input 
                    type="time" 
                    className="input" 
                    value={form.work_end_time} 
                    onChange={e => setForm({ ...form, work_end_time: e.target.value })}
                    required 
                  />
                </div>
                <div className="input-group">
                  <label>Phut canh bao ve som</label>
                  <input 
                    type="number" 
                    className="input" 
                    value={form.allow_early_minutes} 
                    onChange={e => setForm({ ...form, allow_early_minutes: e.target.value })}
                    min={0} 
                    max={180} 
                    required 
                  />
                </div>
              </div>
              
              <small style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '0.5rem', display: 'block' }}>
                Ví dụ: Giờ vào là 08:00 và ân hạn 30 phút, nhân viên điểm danh sau 08:30 sẽ bị tính là Đi muộn (LATE).
              </small>
            </div>
          </div>

          <div>
            {/* Camera Configuration */}
            <div className="card" style={{ marginBottom: '1.25rem' }}>
              <div className="card-header"><span className="card-title">📹 Cấu hình Camera điểm danh</span></div>
              
              <div className="input-group">
                <label>Nguồn camera</label>
                <select 
                  className="input"
                  value={form.camera_source_type}
                  onChange={e => setForm({ ...form, camera_source_type: e.target.value })}
                  style={{ width: '100%' }}
                >
                  <option value="webcam">Webcam máy tính / Thiết bị gắn ngoài</option>
                  <option value="ip_camera">Điện thoại / IP Camera (Dùng địa chỉ URL)</option>
                </select>
              </div>

              {form.camera_source_type === 'webcam' ? (
                <div className="input-group">
                  <label>Số hiệu cổng Webcam (Index)</label>
                  <select 
                    className="input"
                    value={form.camera_webcam_index}
                    onChange={e => setForm({ ...form, camera_webcam_index: e.target.value })}
                    style={{ width: '100%' }}
                  >
                    <option value={0}>Camera 0 (Mặc định)</option>
                    <option value={1}>Camera 1</option>
                    <option value={2}>Camera 2</option>
                    <option value={3}>Camera 3</option>
                  </select>
                  <small style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                    Chỉ số USB webcam kết nối với máy tính
                  </small>
                </div>
              ) : (
                <div className="input-group">
                  <label>Địa chỉ IP Stream (URL)</label>
                  <input 
                    type="text" 
                    className="input" 
                    placeholder="ví dụ: http://192.168.1.5:8080/video"
                    value={form.camera_ip_url} 
                    onChange={e => setForm({ ...form, camera_ip_url: e.target.value })}
                    required
                  />
                  <small style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                    Cổng phát video của app camera điện thoại qua Wi-Fi
                  </small>
                </div>
              )}
            </div>

            {/* Database Info */}
            <div className="card" style={{ marginBottom: '1.25rem' }}>
              <div className="card-header"><span className="card-title">🗄️ Cơ sở dữ liệu</span></div>
              <div style={{ fontSize: '0.85rem', lineHeight: 2.2 }}>
                <div><span style={{ color: 'var(--text-muted)' }}>Nhà cung cấp:</span> <strong>Supabase Cloud</strong></div>
                <div><span style={{ color: 'var(--text-muted)' }}>Khu vực:</span> <strong>Singapore (ap-southeast-1)</strong></div>
                <div><span style={{ color: 'var(--text-muted)' }}>Các bảng:</span> 
                  <span className="badge badge-info" style={{ marginRight: '0.25rem' }}>employees</span>
                  <span className="badge badge-info" style={{ marginRight: '0.25rem' }}>face_embeddings</span>
                  <span className="badge badge-info">attendance_logs</span>
                </div>
              </div>
            </div>

            {/* Model Info */}
            <div className="card">
              <div className="card-header"><span className="card-title">🤖 Mô hình Trí tuệ Nhân tạo</span></div>
              <div style={{ fontSize: '0.85rem', lineHeight: 2.2 }}>
                <div><span style={{ color: 'var(--text-muted)' }}>Mạng trích xuất (Backbone):</span> <strong>ResNet50 + ArcFace</strong></div>
                <div><span style={{ color: 'var(--text-muted)' }}>Kích thước Vector đầu ra:</span> <strong>512 dimensions</strong></div>
                <div><span style={{ color: 'var(--text-muted)' }}>Trọng số nạp (Weights):</span> <strong style={{ color: 'var(--accent-primary)' }}>arcface_vggface2_warmup.pth</strong></div>
                <div><span style={{ color: 'var(--text-muted)' }}>Công cụ dò khuôn mặt:</span> <strong>InsightFace (Buffalo_L)</strong></div>
              </div>
            </div>
          </div>
        </div>

        <div style={{ marginTop: '1.5rem' }}>
          <button type="submit" className="btn btn-primary" style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '0.8rem' }} disabled={saving}>
            <Save size={16} /> {saving ? 'Đang lưu cấu hình...' : 'Lưu cấu hình hệ thống'}
          </button>
        </div>
      </form>
    </div>
  );
}
