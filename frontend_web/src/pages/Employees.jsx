import { useEffect, useState } from 'react';
import { Search, Plus, Pencil, Trash2, X } from 'lucide-react';
import { listEmployees, createEmployee, updateEmployee, deleteEmployee } from '../api';

export default function Employees() {
  const [employees, setEmployees] = useState([]);
  const [search, setSearch] = useState('');
  const [deptFilter, setDeptFilter] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState({ employee_id: '', full_name: '', department: 'IT', position: '', email: '', phone: '' });

  const load = () => {
    listEmployees(search || undefined, deptFilter || undefined).then(r => setEmployees(r.data || []));
  };

  useEffect(() => { load(); }, [search, deptFilter]);

  const handleSave = async () => {
    if (!form.employee_id || !form.full_name) return;
    if (editing) {
      await updateEmployee(editing, form);
    } else {
      await createEmployee(form);
    }
    setShowModal(false);
    setEditing(null);
    setForm({ employee_id: '', full_name: '', department: 'IT', position: '', email: '', phone: '' });
    load();
  };

  const handleEdit = (emp) => {
    setForm(emp);
    setEditing(emp.employee_id);
    setShowModal(true);
  };

  const handleDelete = async (id) => {
    if (confirm(`Xóa nhân viên ${id}?`)) {
      await deleteEmployee(id);
      load();
    }
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Quản lý nhân viên</h1>
        <p>Quản lý thông tin và lý lịch nhân viên</p>
      </div>

      <div className="toolbar">
        <div className="search-bar" style={{ flex: 1, maxWidth: 400 }}>
          <Search size={18} style={{ color: 'var(--text-muted)' }} />
          <input placeholder="Tìm theo mã NV hoặc tên..." value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <select className="input" style={{ width: 160 }} value={deptFilter} onChange={e => setDeptFilter(e.target.value)}>
          <option value="">Tất cả phòng ban</option>
          {['IT', 'HR', 'Finance', 'Marketing', 'Sales', 'Admin'].map(d => <option key={d} value={d}>{d}</option>)}
        </select>
        <div className="toolbar-spacer" />
        <button className="btn btn-primary" onClick={() => { setEditing(null); setForm({ employee_id: '', full_name: '', department: 'IT', position: '', email: '', phone: '' }); setShowModal(true); }}>
          <Plus size={16} /> Thêm NV
        </button>
      </div>

      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: 60 }}>Ảnh</th>
              <th>Mã NV</th>
              <th>Họ tên</th>
              <th>Phòng ban</th>
              <th>Chức vụ</th>
              <th>Email</th>
              <th style={{ textAlign: 'right' }}>Hành động</th>
            </tr>
          </thead>
          <tbody>
            {employees.map(emp => {
              const portraitUrl = `http://localhost:8000/api/portraits/${emp.employee_id}/${emp.employee_id}_000.jpg`;
              return (
                <tr key={emp.employee_id}>
                  <td>
                    <div style={{ width: 40, height: 40, borderRadius: '50%', overflow: 'hidden', border: '1px solid var(--border-color)', background: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <img 
                        src={portraitUrl} 
                        alt={emp.full_name}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(emp.full_name)}&background=random&color=fff&size=100`;
                        }}
                      />
                    </div>
                  </td>
                  <td><strong style={{ color: 'var(--accent-info)' }}>{emp.employee_id}</strong></td>
                  <td>{emp.full_name}</td>
                  <td><span className="badge badge-info">{emp.department}</span></td>
                  <td style={{ color: 'var(--text-secondary)' }}>{emp.position}</td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{emp.email || '—'}</td>
                  <td style={{ textAlign: 'right' }}>
                    <button className="btn-icon" onClick={() => handleEdit(emp)} title="Sửa"><Pencil size={14} /></button>
                    {' '}
                    <button className="btn-icon" onClick={() => handleDelete(emp.employee_id)} title="Xóa" style={{ color: 'var(--accent-danger)' }}><Trash2 size={14} /></button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {employees.length === 0 && (
          <div className="empty-state"><div className="empty-state-icon">👥</div><p>Không tìm thấy nhân viên</p></div>
        )}
      </div>
      <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>Tổng: {employees.length} nhân viên</p>

      {/* Modal */}
      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div className="modal-title">{editing ? 'Sửa Nhân Viên' : 'Thêm Nhân Viên Mới'}</div>
              <button className="btn-icon" onClick={() => setShowModal(false)}><X size={16} /></button>
            </div>
            
            {/* Employee Portrait Preview inside Modal */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', margin: '0.5rem 0 1.5rem 0' }}>
              <div style={{ width: 90, height: 90, borderRadius: '50%', overflow: 'hidden', border: '2px solid var(--accent-primary)', background: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: 'var(--shadow-md)' }}>
                <img 
                  src={`http://localhost:8000/api/portraits/${form.employee_id}/${form.employee_id}_000.jpg`} 
                  alt="Portrait"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(form.full_name || 'User')}&background=e2e8f0&color=475569&size=200`;
                  }}
                />
              </div>
              {form.employee_id && (
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  Ảnh chân dung: {form.employee_id}
                </span>
              )}
            </div>

            <div className="input-group">
              <label>Mã NV *</label>
              <input className="input" value={form.employee_id} onChange={e => setForm({ ...form, employee_id: e.target.value })} disabled={!!editing} placeholder="NV001" />
            </div>
            <div className="input-group">
              <label>Họ tên *</label>
              <input className="input" value={form.full_name} onChange={e => setForm({ ...form, full_name: e.target.value })} placeholder="Nguyễn Văn A" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div className="input-group">
                <label>Phòng ban</label>
                <select className="input" value={form.department} onChange={e => setForm({ ...form, department: e.target.value })}>
                  {['IT', 'HR', 'Finance', 'Marketing', 'Sales', 'Admin'].map(d => <option key={d}>{d}</option>)}
                </select>
              </div>
              <div className="input-group">
                <label>Chức vụ</label>
                <input className="input" value={form.position} onChange={e => setForm({ ...form, position: e.target.value })} placeholder="Developer" />
              </div>
            </div>
            <div className="input-group">
              <label>Email</label>
              <input className="input" value={form.email || ''} onChange={e => setForm({ ...form, email: e.target.value })} placeholder="email@company.com" />
            </div>
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
