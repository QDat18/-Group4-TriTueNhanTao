import { useEffect, useState } from 'react';
import { Plus, Power, Trash2, X } from 'lucide-react';
import { listDevices, createDevice, toggleDevice, deleteDevice } from '../api';

export default function Cameras() {
  const [devices, setDevices] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState({ device_id: '', device_name: '', location: '' });

  const load = () => listDevices().then(r => setDevices(r.data || []));
  useEffect(() => { load(); }, []);

  const handleSave = async () => {
    if (!form.device_id || !form.device_name) return;
    await createDevice({ ...form, is_active: true });
    setShowModal(false);
    setForm({ device_id: '', device_name: '', location: '' });
    load();
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Camera Management</h1>
        <p>Quản lý camera chấm công</p>
      </div>

      <div className="toolbar">
        <div className="toolbar-spacer" />
        <button className="btn btn-primary" onClick={() => setShowModal(true)}><Plus size={16} /> Thêm Camera</button>
      </div>

      <div className="grid-3">
        {devices.map(dev => (
          <div className="card" key={dev.device_id} style={{ position: 'relative' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: dev.is_active ? '#34d399' : '#f87171', boxShadow: dev.is_active ? '0 0 8px #34d399' : 'none' }} />
              <div>
                <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>{dev.device_id}</div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{dev.device_name}</div>
              </div>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>📍 {dev.location || 'Chưa có vị trí'}</p>
            <span className={`badge ${dev.is_active ? 'badge-success' : 'badge-danger'}`}>
              {dev.is_active ? '● Online' : '● Offline'}
            </span>
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
              <button className="btn btn-sm btn-secondary" style={{ flex: 1 }} onClick={() => { toggleDevice(dev.device_id).then(load); }}>
                <Power size={14} /> {dev.is_active ? 'Tắt' : 'Bật'}
              </button>
              <button className="btn btn-sm btn-danger" onClick={() => { if (confirm('Xóa?')) deleteDevice(dev.device_id).then(load); }}>
                <Trash2 size={14} />
              </button>
            </div>
          </div>
        ))}
      </div>

      {devices.length === 0 && <div className="card"><div className="empty-state"><div className="empty-state-icon">🎥</div><p>Chưa có camera. Hãy thêm camera mới.</p></div></div>}

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="modal-title">Thêm Camera Mới</div>
              <button className="btn-icon" onClick={() => setShowModal(false)}><X size={16} /></button>
            </div>
            <div className="input-group"><label>Camera ID *</label><input className="input" value={form.device_id} onChange={e => setForm({ ...form, device_id: e.target.value })} placeholder="CAM001" /></div>
            <div className="input-group"><label>Tên Camera *</label><input className="input" value={form.device_name} onChange={e => setForm({ ...form, device_name: e.target.value })} placeholder="Camera Tầng 1" /></div>
            <div className="input-group"><label>Vị trí</label><input className="input" value={form.location} onChange={e => setForm({ ...form, location: e.target.value })} placeholder="Tầng 1 - Sảnh chính" /></div>
            <div className="modal-actions">
              <button className="btn btn-secondary" onClick={() => setShowModal(false)}>Hủy</button>
              <button className="btn btn-primary" onClick={handleSave}>💾 Lưu</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
