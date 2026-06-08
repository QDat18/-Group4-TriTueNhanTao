import { useEffect, useState } from 'react';
import { Search, Plus, Pencil, Trash2, X } from 'lucide-react';
import { listEmployees, createEmployee, updateEmployee, deleteEmployee } from '../api';

export default function Employees() {
  const [employees, setEmployees] = useState([]);
  const [search, setSearch] = useState('');
  const [deptFilter, setDeptFilter] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState({ employee_id:'', full_name:'', department:'IT', position:'', email:'', phone:'' });

  const load = () => listEmployees(search||undefined, deptFilter||undefined).then(r => setEmployees(r.data||[]));
  useEffect(() => { load(); }, [search, deptFilter]);

  const handleSave = async () => {
    if (!form.employee_id || !form.full_name) return;
    editing ? await updateEmployee(editing, form) : await createEmployee(form);
    setShowModal(false); setEditing(null);
    setForm({ employee_id:'', full_name:'', department:'IT', position:'', email:'', phone:'' });
    load();
  };
  const handleEdit = (emp) => { setForm(emp); setEditing(emp.employee_id); setShowModal(true); };
  const handleDelete = async (id) => { if (confirm(`Xóa nhân viên ${id}?`)) { await deleteEmployee(id); load(); } };

  const LBL = ({ children }) => (
    <label style={{ display:'block', fontSize:'0.68rem', fontWeight:700, color:'#6b7280', textTransform:'uppercase', letterSpacing:'0.05em', marginBottom:'0.25rem' }}>
      {children}
    </label>
  );
  const inputSt = { padding:'0.5rem 0.75rem', fontSize:'0.85rem' };
  const fieldSt = { marginBottom:'0.625rem' };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Quản lý nhân viên</h1>
        <p>Quản lý thông tin và lý lịch nhân viên</p>
      </div>

      <div className="toolbar">
        <div className="search-bar" style={{ flex:1, maxWidth:400 }}>
          <Search size={16} style={{ color:'#9ca3af', flexShrink:0 }} />
          <input placeholder="Tìm theo mã NV hoặc tên..." value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <select className="input" style={{ width:160 }} value={deptFilter} onChange={e => setDeptFilter(e.target.value)}>
          <option value="">Tất cả phòng ban</option>
          {['IT','HR','Finance','Marketing','Sales','Admin'].map(d => <option key={d} value={d}>{d}</option>)}
        </select>
        <div className="toolbar-spacer" />
        <button className="btn btn-primary" onClick={() => { setEditing(null); setForm({ employee_id:'', full_name:'', department:'IT', position:'', email:'', phone:'' }); setShowModal(true); }}>
          <Plus size={15} /> Thêm NV
        </button>
      </div>

      <div className="card" style={{ padding:0, overflow:'hidden' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width:56 }}>Ảnh</th>
              <th>Mã NV</th><th>Họ tên</th><th>Phòng ban</th><th>Chức vụ</th><th>Email</th>
              <th style={{ textAlign:'right', paddingRight:'1.25rem' }}>Hành động</th>
            </tr>
          </thead>
          <tbody>
            {employees.map(emp => (
              <tr key={emp.employee_id}>
                <td>
                  <div style={{ width:36, height:36, borderRadius:'50%', overflow:'hidden', border:'2px solid #e5e7eb', background:'#f3f4f6' }}>
                    <img src={`http://localhost:8000/api/portraits/${emp.employee_id}/${emp.employee_id}_000.jpg`} alt={emp.full_name}
                      style={{ width:'100%', height:'100%', objectFit:'cover' }}
                      onError={e => { e.target.onerror=null; e.target.src=`https://ui-avatars.com/api/?name=${encodeURIComponent(emp.full_name)}&background=dbeafe&color=1d4ed8&size=80`; }}
                    />
                  </div>
                </td>
                <td><strong style={{ color:'#2563eb', fontSize:'0.85rem' }}>{emp.employee_id}</strong></td>
                <td style={{ fontWeight:500, color:'#111827' }}>{emp.full_name}</td>
                <td><span className="badge badge-info">{emp.department}</span></td>
                <td style={{ color:'#6b7280', fontSize:'0.85rem' }}>{emp.position}</td>
                <td style={{ color:'#9ca3af', fontSize:'0.8rem' }}>{emp.email||'—'}</td>
                <td style={{ textAlign:'right', paddingRight:'1rem' }}>
                  <button className="btn-icon" onClick={() => handleEdit(emp)} style={{ marginRight:4 }}><Pencil size={13} /></button>
                  <button className="btn-icon" onClick={() => handleDelete(emp.employee_id)} style={{ color:'#dc2626' }}><Trash2 size={13} /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {employees.length === 0 && <div className="empty-state"><div className="empty-state-icon">👥</div><p>Không tìm thấy nhân viên</p></div>}
      </div>
      <p style={{ fontSize:'0.775rem', color:'#9ca3af', marginTop:'0.75rem' }}>Tổng: {employees.length} nhân viên</p>

      {showModal && (
        <div onClick={() => setShowModal(false)} style={{
          position:'fixed', inset:0, zIndex:200,
          background:'rgba(17,24,39,.5)',
          backdropFilter:'blur(6px)',
          WebkitBackdropFilter:'blur(6px)',
          display:'flex', alignItems:'center', justifyContent:'center',
          padding:'1rem',
        }}>
          <div onClick={e => e.stopPropagation()} style={{
            background:'#fff', borderRadius:14,
            border:'1px solid #e5e7eb',
            boxShadow:'0 24px 60px rgba(0,0,0,.22)',
            width:'100%', maxWidth:400,
          }}>
            {/* Header */}
            <div style={{ padding:'0.875rem 1rem', borderBottom:'1px solid #f3f4f6', display:'flex', justifyContent:'space-between', alignItems:'center' }}>
              <div style={{ fontSize:'0.95rem', fontWeight:800, color:'#111827' }}>
                {editing ? 'Sửa Nhân Viên' : 'Thêm Nhân Viên Mới'}
              </div>
              <button className="btn-icon" style={{ width:28, height:28 }} onClick={() => setShowModal(false)}><X size={14} /></button>
            </div>

            {/* Body */}
            <div style={{ padding:'0.875rem 1rem' }}>
              {/* Avatar */}
              <div style={{ display:'flex', alignItems:'center', gap:'0.625rem', padding:'0.5rem 0.75rem', background:'#f8fafc', borderRadius:8, border:'1px solid #e5e7eb', marginBottom:'0.875rem' }}>
                <div style={{ width:38, height:38, borderRadius:'50%', overflow:'hidden', border:'2px solid #dbeafe', flexShrink:0 }}>
                  <img src={`http://localhost:8000/api/portraits/${form.employee_id}/${form.employee_id}_000.jpg`} alt="Portrait"
                    style={{ width:'100%', height:'100%', objectFit:'cover' }}
                    onError={e => { e.target.onerror=null; e.target.src=`https://ui-avatars.com/api/?name=${encodeURIComponent(form.full_name||'User')}&background=dbeafe&color=1d4ed8&size=200`; }}
                  />
                </div>
                <div>
                  <div style={{ fontWeight:700, fontSize:'0.825rem', color:'#111827' }}>{form.full_name||'Nhân viên mới'}</div>
                  <div style={{ fontSize:'0.68rem', color:'#9ca3af' }}>{form.employee_id||'Chưa có mã'}</div>
                </div>
              </div>

              <div style={fieldSt}><LBL>Mã NV *</LBL>
                <input className="input" style={inputSt} value={form.employee_id} onChange={e => setForm({...form, employee_id:e.target.value})} disabled={!!editing} placeholder="NV001" />
              </div>
              <div style={fieldSt}><LBL>Họ tên *</LBL>
                <input className="input" style={inputSt} value={form.full_name} onChange={e => setForm({...form, full_name:e.target.value})} placeholder="Nguyễn Văn A" />
              </div>
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'0.625rem', marginBottom:'0.625rem' }}>
                <div><LBL>Phòng ban</LBL>
                  <select className="input" style={inputSt} value={form.department} onChange={e => setForm({...form, department:e.target.value})}>
                    {['IT','HR','Finance','Marketing','Sales','Admin'].map(d => <option key={d}>{d}</option>)}
                  </select>
                </div>
                <div><LBL>Chức vụ</LBL>
                  <input className="input" style={inputSt} value={form.position} onChange={e => setForm({...form, position:e.target.value})} placeholder="Developer" />
                </div>
              </div>
              <div><LBL>Email</LBL>
                <input className="input" style={inputSt} value={form.email||''} onChange={e => setForm({...form, email:e.target.value})} placeholder="email@company.com" />
              </div>
            </div>

            {/* Footer */}
            <div style={{ padding:'0.625rem 1rem', borderTop:'1px solid #f3f4f6', display:'flex', gap:'0.5rem', justifyContent:'flex-end' }}>
              <button className="btn btn-secondary btn-sm" onClick={() => setShowModal(false)}>Hủy</button>
              <button className="btn btn-primary btn-sm" onClick={handleSave}>💾 Lưu</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}