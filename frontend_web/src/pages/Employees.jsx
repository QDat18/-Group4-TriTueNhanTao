import { useEffect, useMemo, useState } from 'react';
import {
  Search,
  Plus,
  Pencil,
  Trash2,
  X,
  Mail,
  Phone,
  Building2,
  Briefcase,
  Users,
  BadgeCheck,
  RefreshCw
} from 'lucide-react';

import {
  listEmployees,
  createEmployee,
  updateEmployee,
  deleteEmployee
} from '../api';

const API_BASE = 'http://localhost:8000';

const departments = [
  'IT',
  'HR',
  'Finance',
  'Marketing',
  'Sales',
  'Admin'
];

function getNextEmployeeId(employees) {
  const usedNumbers = employees
    .map(emp => String(emp.employee_id || '').trim())
    .map(id => {
      const match = id.match(/^NV(\d+)$/i);
      return match ? Number(match[1]) : null;
    })
    .filter(num => Number.isInteger(num));

  const nextNumber = usedNumbers.length > 0 ? Math.max(...usedNumbers) + 1 : 1;
  return `NV${String(nextNumber).padStart(3, '0')}`;
}

async function deleteEmployeeAssets(employeeId) {
  const endpoints = [
    `${API_BASE}/api/embeddings/${encodeURIComponent(employeeId)}`,
    `${API_BASE}/api/portraits/${encodeURIComponent(employeeId)}`,
    `${API_BASE}/api/employees/${encodeURIComponent(employeeId)}/embedding`
  ];

  const results = [];

  for (const url of endpoints) {
    try {
      const res = await fetch(url, { method: 'DELETE' });
      results.push({ url, ok: res.ok, status: res.status });
    } catch (err) {
      results.push({ url, ok: false, error: err.message });
    }
  }

  return results;
}


function Avatar({ employeeId, fullName, size = 48 }) {
  const name = fullName || 'User';

  return (
    <div style={{
      width: size,
      height: size,
      borderRadius: size >= 72 ? 20 : '50%',
      overflow: 'hidden',
      border: '2px solid #e5e7eb',
      background: 'linear-gradient(135deg,#eff6ff,#dbeafe)',
      flexShrink: 0,
      boxShadow: size >= 72 ? '0 10px 25px rgba(37,99,235,.12)' : 'none'
    }}>
      <img
        src={`${API_BASE}/api/portraits/${employeeId}/portrait.jpg`}
        alt={name}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          display: 'block'
        }}
        onError={e => {
          e.target.onerror = null;
          e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(name)}&background=dbeafe&color=1d4ed8&size=160&bold=true`;
        }}
      />
    </div>
  );
}

function StatPill({ icon, label, value, color }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.7rem',
      background: '#fff',
      border: '1px solid #e5e7eb',
      borderRadius: 16,
      padding: '0.9rem 1rem',
      boxShadow: '0 1px 3px rgba(0,0,0,.06)'
    }}>
      <div style={{
        width: 38,
        height: 38,
        borderRadius: 12,
        background: color.bg,
        color: color.text,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        {icon}
      </div>

      <div>
        <div style={{
          fontSize: '1.25rem',
          fontWeight: 850,
          color: '#111827',
          lineHeight: 1
        }}>
          {value}
        </div>

        <div style={{
          fontSize: '0.72rem',
          color: '#6b7280',
          marginTop: 4,
          fontWeight: 600
        }}>
          {label}
        </div>
      </div>
    </div>
  );
}

function EmployeeCard({ emp, onEdit, onDelete }) {
  return (
    <div style={{
      background: '#fff',
      border: '1px solid #e5e7eb',
      borderRadius: 18,
      overflow: 'hidden',
      boxShadow: '0 1px 3px rgba(0,0,0,.06)',
      transition: 'transform 180ms ease, box-shadow 180ms ease'
    }}
      onMouseEnter={e => {
        e.currentTarget.style.transform = 'translateY(-3px)';
        e.currentTarget.style.boxShadow = '0 12px 30px rgba(15,23,42,.10)';
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,.06)';
      }}
    >
      <div style={{
        height: 70,
        background: 'linear-gradient(135deg,#2563eb,#7c3aed)',
        position: 'relative'
      }}>
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'radial-gradient(circle at 80% 20%, rgba(255,255,255,.35), transparent 28%)'
        }} />

        <div style={{
          position: 'absolute',
          left: 18,
          bottom: -34
        }}>
          <Avatar
            employeeId={emp.employee_id}
            fullName={emp.full_name}
            size={72}
          />
        </div>

        <div style={{
          position: 'absolute',
          right: 14,
          top: 14,
          display: 'flex',
          gap: '0.45rem'
        }}>
          <button
            className="btn-icon"
            onClick={() => onEdit(emp)}
            title="Sửa"
            style={{
              background: 'rgba(255,255,255,.92)',
              color: '#2563eb'
            }}
          >
            <Pencil size={14} />
          </button>

          <button
            className="btn-icon"
            onClick={() => onDelete(emp.employee_id)}
            title="Xóa"
            style={{
              background: 'rgba(255,255,255,.92)',
              color: '#dc2626'
            }}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      <div style={{
        padding: '2.65rem 1.1rem 1rem'
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          gap: '0.75rem',
          alignItems: 'flex-start'
        }}>
          <div style={{ minWidth: 0 }}>
            <div style={{
              fontWeight: 850,
              color: '#111827',
              fontSize: '1rem',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}>
              {emp.full_name}
            </div>

            <div style={{
              fontSize: '0.78rem',
              color: '#2563eb',
              fontWeight: 750,
              marginTop: 2
            }}>
              {emp.employee_id}
            </div>
          </div>

          <span style={{
            fontSize: '0.68rem',
            color: '#065f46',
            background: '#ecfdf5',
            border: '1px solid #a7f3d0',
            borderRadius: 999,
            padding: '0.2rem 0.5rem',
            fontWeight: 800,
            flexShrink: 0
          }}>
            Active
          </span>
        </div>

        <div style={{
          marginTop: '1rem',
          display: 'grid',
          gap: '0.55rem'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.55rem',
            color: '#475569',
            fontSize: '0.78rem'
          }}>
            <Building2 size={14} style={{ color: '#2563eb' }} />
            <span className="badge badge-info">{emp.department || '—'}</span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.55rem',
            color: '#64748b',
            fontSize: '0.78rem'
          }}>
            <Briefcase size={14} style={{ color: '#64748b' }} />
            <span>{emp.position || 'Chưa cập nhật chức vụ'}</span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.55rem',
            color: '#94a3b8',
            fontSize: '0.76rem',
            minHeight: 18
          }}>
            <Mail size={14} />
            <span style={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}>
              {emp.email || 'Chưa có email'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Employees() {
  const [employees, setEmployees] = useState([]);
  const [search, setSearch] = useState('');
  const [deptFilter, setDeptFilter] = useState('');
  const [viewMode, setViewMode] = useState('cards');
  const [showModal, setShowModal] = useState(false);
  const [editing, setEditing] = useState(null);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);

  const [form, setForm] = useState({
    employee_id: '',
    full_name: '',
    department: 'IT',
    position: '',
    email: '',
    phone: ''
  });

  const load = () => {
    setLoading(true);

    listEmployees(search || undefined, deptFilter || undefined)
      .then(res => {
        if (res?.error) {
          throw new Error(res.error);
        }

        setEmployees(res.data || []);
      })
      .catch(err => {
        setNotice({
          type: 'error',
          message: `Không tải được danh sách nhân viên: ${err.message || 'Lỗi không xác định'}`
        });
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    load();
  }, [search, deptFilter]);

  const departmentCounts = useMemo(() => {
    return employees.reduce((acc, emp) => {
      const dept = emp.department || 'Khác';
      acc[dept] = (acc[dept] || 0) + 1;
      return acc;
    }, {});
  }, [employees]);

  const handleOpenCreate = () => {
    const suggestedId = getNextEmployeeId(employees);

    setEditing(null);
    setForm({
      employee_id: suggestedId,
      full_name: '',
      department: 'IT',
      position: '',
      email: '',
      phone: ''
    });
    setShowModal(true);
  };

  const handleSave = async () => {
    const payload = {
      ...form,
      employee_id: form.employee_id.trim(),
      full_name: form.full_name.trim(),
      email: form.email?.trim() || '',
      phone: form.phone?.trim() || ''
    };

    if (!payload.employee_id || !payload.full_name) {
      setNotice({
        type: 'error',
        message: 'Vui lòng nhập đầy đủ Mã NV và Họ tên.'
      });
      return;
    }

    setLoading(true);
    setNotice(null);

    try {
      const res = editing
        ? await updateEmployee(editing, payload)
        : await createEmployee(payload);

      if (res?.error) {
        throw new Error(res.error);
      }

      setNotice({
        type: 'success',
        message: editing
          ? `Đã cập nhật nhân viên ${payload.employee_id} thành công.`
          : `Đã thêm nhân viên ${payload.employee_id} thành công.`
      });

      setShowModal(false);
      setEditing(null);
      load();
    } catch (err) {
      setNotice({
        type: 'error',
        message: `Không lưu được nhân viên: ${err.message || 'Kiểm tra API/backend hoặc mã nhân viên đã tồn tại.'}`
      });
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = (emp) => {
    setForm({
      employee_id: emp.employee_id || '',
      full_name: emp.full_name || '',
      department: emp.department || 'IT',
      position: emp.position || '',
      email: emp.email || '',
      phone: emp.phone || ''
    });

    setEditing(emp.employee_id);
    setShowModal(true);
  };

  const handleDelete = (id) => {
    const emp = employees.find(item => item.employee_id === id);
    setDeleteTarget(emp || { employee_id: id });
  };

  const confirmDelete = async () => {
    if (!deleteTarget?.employee_id) {
      return;
    }

    const id = deleteTarget.employee_id;

    setLoading(true);
    setNotice(null);

    try {
      await deleteEmployeeAssets(id);

      const res = await deleteEmployee(id);

      if (res?.error) {
        throw new Error(res.error);
      }

      setNotice({
        type: 'success',
        message: `Đã xóa nhân viên ${id}, kèm embedding và ảnh mẫu nếu backend có endpoint hỗ trợ.`
      });
      setDeleteTarget(null);
      load();
    } catch (err) {
      setNotice({
        type: 'error',
        message: `Không xóa được nhân viên ${id}: ${err.message || 'Lỗi không xác định'}`
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-in">
      <div style={{
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'space-between',
        marginBottom: '1.4rem',
        gap: '1rem'
      }}>
        <div>
          <h1 style={{
            fontSize: '1.75rem',
            fontWeight: 850,
            color: '#111827',
            margin: 0,
            letterSpacing: '-0.03em'
          }}>
            Quản lý nhân viên
          </h1>

          <p style={{
            color: '#6b7280',
            fontSize: '0.9rem',
            marginTop: '0.3rem'
          }}>
            Quản lý hồ sơ, phòng ban và ảnh đại diện nhân viên.
          </p>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleOpenCreate}
          style={{
            boxShadow: '0 8px 18px rgba(37,99,235,.22)'
          }}
        >
          <Plus size={15} />
          Thêm nhân viên
        </button>
      </div>

      {notice && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.55rem',
          marginBottom: '1rem',
          padding: '0.75rem 0.95rem',
          borderRadius: 12,
          border: `1px solid ${notice.type === 'success' ? '#bbf7d0' : '#fecaca'}`,
          background: notice.type === 'success' ? '#f0fdf4' : '#fef2f2',
          color: notice.type === 'success' ? '#15803d' : '#b91c1c',
          fontSize: '0.82rem',
          fontWeight: 650
        }}>
          {notice.type === 'success' ? '✅' : '❌'} {notice.message}
        </div>
      )}

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
        gap: '1rem',
        marginBottom: '1.25rem'
      }}>
        <StatPill
          icon={<Users size={19} />}
          label="Tổng nhân viên"
          value={employees.length}
          color={{ bg: '#eff6ff', text: '#2563eb' }}
        />

        <StatPill
          icon={<Building2 size={19} />}
          label="Phòng ban đang hiển thị"
          value={Object.keys(departmentCounts).length}
          color={{ bg: '#f0fdf4', text: '#16a34a' }}
        />

        <StatPill
          icon={<BadgeCheck size={19} />}
          label="Hồ sơ hợp lệ"
          value={employees.filter(e => e.employee_id && e.full_name).length}
          color={{ bg: '#fef3c7', text: '#d97706' }}
        />
      </div>

      <div style={{
        background: '#fff',
        border: '1px solid #e5e7eb',
        borderRadius: 16,
        padding: '0.9rem',
        display: 'flex',
        gap: '0.75rem',
        alignItems: 'center',
        marginBottom: '1.25rem',
        boxShadow: '0 1px 3px rgba(0,0,0,.05)'
      }}>
        <div className="search-bar" style={{
          flex: 1,
          maxWidth: 460,
          background: '#f9fafb'
        }}>
          <Search
            size={16}
            style={{
              color: '#9ca3af',
              flexShrink: 0
            }}
          />

          <input
            placeholder="Tìm theo mã nhân viên hoặc họ tên..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>

        <select
          className="input"
          style={{ width: 170 }}
          value={deptFilter}
          onChange={e => setDeptFilter(e.target.value)}
        >
          <option value="">Tất cả phòng ban</option>

          {departments.map(dept => (
            <option
              key={dept}
              value={dept}
            >
              {dept}
            </option>
          ))}
        </select>

        <div style={{
          display: 'flex',
          background: '#f3f4f6',
          borderRadius: 10,
          padding: 3,
          border: '1px solid #e5e7eb'
        }}>
          <button
            onClick={() => setViewMode('cards')}
            style={{
              border: 'none',
              background: viewMode === 'cards' ? '#fff' : 'transparent',
              padding: '0.45rem 0.7rem',
              borderRadius: 8,
              fontSize: '0.75rem',
              fontWeight: 750,
              color: viewMode === 'cards' ? '#2563eb' : '#6b7280',
              boxShadow: viewMode === 'cards' ? '0 1px 3px rgba(0,0,0,.08)' : 'none',
              cursor: 'pointer'
            }}
          >
            Card
          </button>

          <button
            onClick={() => setViewMode('table')}
            style={{
              border: 'none',
              background: viewMode === 'table' ? '#fff' : 'transparent',
              padding: '0.45rem 0.7rem',
              borderRadius: 8,
              fontSize: '0.75rem',
              fontWeight: 750,
              color: viewMode === 'table' ? '#2563eb' : '#6b7280',
              boxShadow: viewMode === 'table' ? '0 1px 3px rgba(0,0,0,.08)' : 'none',
              cursor: 'pointer'
            }}
          >
            Bảng
          </button>
        </div>

        <button
          className="btn btn-secondary"
          onClick={load}
          disabled={loading}
        >
          <RefreshCw
            size={14}
            style={{
              animation: loading ? 'spin 1s linear infinite' : 'none'
            }}
          />
          Làm mới
        </button>
      </div>

      {viewMode === 'cards' ? (
        <>
          {employees.length > 0 ? (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(255px, 1fr))',
              gap: '1rem'
            }}>
              {employees.map(emp => (
                <EmployeeCard
                  key={emp.employee_id}
                  emp={emp}
                  onEdit={handleEdit}
                  onDelete={handleDelete}
                />
              ))}
            </div>
          ) : (
            <div className="card">
              <div className="empty-state">
                <div className="empty-state-icon">👥</div>
                <p>Không tìm thấy nhân viên.</p>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="card" style={{
          padding: 0,
          overflow: 'hidden'
        }}>
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 70 }}>Ảnh</th>
                <th>Mã NV</th>
                <th>Họ tên</th>
                <th>Phòng ban</th>
                <th>Chức vụ</th>
                <th>Email</th>
                <th style={{
                  textAlign: 'right',
                  paddingRight: '1.25rem'
                }}>
                  Hành động
                </th>
              </tr>
            </thead>

            <tbody>
              {employees.map(emp => (
                <tr key={emp.employee_id}>
                  <td>
                    <Avatar
                      employeeId={emp.employee_id}
                      fullName={emp.full_name}
                      size={42}
                    />
                  </td>

                  <td>
                    <strong style={{
                      color: '#2563eb',
                      fontSize: '0.85rem'
                    }}>
                      {emp.employee_id}
                    </strong>
                  </td>

                  <td style={{
                    fontWeight: 650,
                    color: '#111827'
                  }}>
                    {emp.full_name}
                  </td>

                  <td>
                    <span className="badge badge-info">
                      {emp.department || '—'}
                    </span>
                  </td>

                  <td style={{
                    color: '#6b7280',
                    fontSize: '0.85rem'
                  }}>
                    {emp.position || '—'}
                  </td>

                  <td style={{
                    color: '#9ca3af',
                    fontSize: '0.8rem'
                  }}>
                    {emp.email || '—'}
                  </td>

                  <td style={{
                    textAlign: 'right',
                    paddingRight: '1rem'
                  }}>
                    <button
                      className="btn-icon"
                      onClick={() => handleEdit(emp)}
                      style={{ marginRight: 4 }}
                    >
                      <Pencil size={13} />
                    </button>

                    <button
                      className="btn-icon"
                      onClick={() => handleDelete(emp.employee_id)}
                      style={{ color: '#dc2626' }}
                    >
                      <Trash2 size={13} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {employees.length === 0 && (
            <div className="empty-state">
              <div className="empty-state-icon">👥</div>
              <p>Không tìm thấy nhân viên.</p>
            </div>
          )}
        </div>
      )}

      <p style={{
        fontSize: '0.78rem',
        color: '#9ca3af',
        marginTop: '0.9rem'
      }}>
        Tổng: {employees.length} nhân viên
      </p>

      {showModal && (
        <div
          onClick={() => setShowModal(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15,23,42,.58)',
            backdropFilter: 'blur(8px)',
            WebkitBackdropFilter: 'blur(8px)',
            zIndex: 200,
            overflowY: 'auto'
          }}
        >
          <div style={{
            minHeight: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1rem'
          }}>
            <div
              onClick={e => e.stopPropagation()}
              style={{
                background: '#fff',
                borderRadius: 18,
                border: '1px solid #e5e7eb',
                boxShadow: '0 25px 80px rgba(15,23,42,.25)',
                width: '100%',
                maxWidth: 520,
                overflow: 'hidden'
              }}
            >
              <div style={{
                background: 'linear-gradient(135deg,#2563eb,#7c3aed)',
                padding: '1.2rem 1.35rem',
                color: '#fff',
                position: 'relative'
              }}>
                <button
                  className="btn-icon"
                  onClick={() => setShowModal(false)}
                  style={{
                    position: 'absolute',
                    right: 14,
                    top: 14,
                    background: 'rgba(255,255,255,.14)',
                    color: '#fff'
                  }}
                >
                  <X size={16} />
                </button>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.9rem'
                }}>
                  <Avatar
                    employeeId={form.employee_id}
                    fullName={form.full_name || 'User'}
                    size={70}
                  />

                  <div>
                    <div style={{
                      fontSize: '1.05rem',
                      fontWeight: 850,
                      letterSpacing: '-0.02em'
                    }}>
                      {editing ? 'Cập nhật nhân viên' : 'Thêm nhân viên mới'}
                    </div>

                    <div style={{
                      fontSize: '0.78rem',
                      color: 'rgba(255,255,255,.78)',
                      marginTop: 3
                    }}>
                      {form.employee_id ? `Mã nhân viên: ${form.employee_id}` : 'Tạo hồ sơ nhân viên mới'}
                    </div>
                  </div>
                </div>
              </div>

              <div style={{
                padding: '1.25rem'
              }}>
                <div className="input-group">
                  <label>Mã NV *</label>
                  <input
                    className="input"
                    value={form.employee_id}
                    onChange={e => setForm({
                      ...form,
                      employee_id: e.target.value
                    })}
                    disabled={!!editing}
                    placeholder="NV001"
                  />
                  {!editing && (
                    <div style={{
                      marginTop: 6,
                      fontSize: '0.72rem',
                      color: '#2563eb',
                      fontWeight: 700
                    }}>
                      Gợi ý mã tiếp theo: {getNextEmployeeId(employees)}
                    </div>
                  )}
                </div>

                <div className="input-group">
                  <label>Họ tên *</label>
                  <input
                    className="input"
                    value={form.full_name}
                    onChange={e => setForm({
                      ...form,
                      full_name: e.target.value
                    })}
                    placeholder="Nguyễn Văn A"
                  />
                </div>

                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '0.85rem'
                }}>
                  <div className="input-group">
                    <label>Phòng ban</label>
                    <select
                      className="input"
                      value={form.department}
                      onChange={e => setForm({
                        ...form,
                        department: e.target.value
                      })}
                    >
                      {departments.map(dept => (
                        <option key={dept}>
                          {dept}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="input-group">
                    <label>Chức vụ</label>
                    <input
                      className="input"
                      value={form.position}
                      onChange={e => setForm({
                        ...form,
                        position: e.target.value
                      })}
                      placeholder="Developer"
                    />
                  </div>
                </div>

                <div className="input-group">
                  <label>Email</label>
                  <div style={{
                    position: 'relative'
                  }}>
                    <Mail
                      size={15}
                      style={{
                        position: 'absolute',
                        left: 12,
                        top: '50%',
                        transform: 'translateY(-50%)',
                        color: '#9ca3af'
                      }}
                    />
                    <input
                      className="input"
                      value={form.email || ''}
                      onChange={e => setForm({
                        ...form,
                        email: e.target.value
                      })}
                      placeholder="email@company.com"
                      style={{ paddingLeft: 36 }}
                    />
                  </div>
                </div>

                <div className="input-group" style={{ marginBottom: 0 }}>
                  <label>Số điện thoại</label>
                  <div style={{
                    position: 'relative'
                  }}>
                    <Phone
                      size={15}
                      style={{
                        position: 'absolute',
                        left: 12,
                        top: '50%',
                        transform: 'translateY(-50%)',
                        color: '#9ca3af'
                      }}
                    />
                    <input
                      className="input"
                      value={form.phone || ''}
                      onChange={e => setForm({
                        ...form,
                        phone: e.target.value
                      })}
                      placeholder="090..."
                      style={{ paddingLeft: 36 }}
                    />
                  </div>
                </div>
              </div>

              <div style={{
                padding: '0.95rem 1.25rem',
                borderTop: '1px solid #f3f4f6',
                background: '#f8fafc',
                display: 'flex',
                justifyContent: 'flex-end',
                gap: '0.7rem'
              }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => setShowModal(false)}
                >
                  Hủy
                </button>

                <button
                  className="btn btn-primary"
                  onClick={handleSave}
                  disabled={loading || !form.employee_id || !form.full_name}
                >
                  {loading ? '⏳ Đang lưu...' : '💾 Lưu'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {deleteTarget && (
        <div
          onClick={() => !loading && setDeleteTarget(null)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15,23,42,.58)',
            backdropFilter: 'blur(8px)',
            WebkitBackdropFilter: 'blur(8px)',
            zIndex: 260,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1rem'
          }}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{
              width: '100%',
              maxWidth: 470,
              background: '#fff',
              borderRadius: 18,
              border: '1px solid #fecaca',
              boxShadow: '0 25px 80px rgba(15,23,42,.25)',
              overflow: 'hidden'
            }}
          >
            <div style={{
              padding: '1.1rem 1.25rem',
              background: '#fef2f2',
              borderBottom: '1px solid #fecaca',
              display: 'flex',
              gap: '0.8rem',
              alignItems: 'center'
            }}>
              <div style={{
                width: 42,
                height: 42,
                borderRadius: 14,
                background: '#fee2e2',
                color: '#dc2626',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0
              }}>
                <Trash2 size={20} />
              </div>
              <div>
                <div style={{ fontWeight: 850, color: '#991b1b', fontSize: '1rem' }}>
                  Xác nhận xóa nhân viên
                </div>
                <div style={{ color: '#b91c1c', fontSize: '0.78rem', marginTop: 3 }}>
                  Thao tác này sẽ xóa hồ sơ, embedding và ảnh mẫu liên quan nếu backend đã có endpoint.
                </div>
              </div>
            </div>

            <div style={{ padding: '1.2rem 1.25rem' }}>
              <p style={{ margin: 0, color: '#374151', fontSize: '0.9rem', lineHeight: 1.6 }}>
                Bạn chắc chắn muốn xóa nhân viên
                {' '}
                <strong style={{ color: '#111827' }}>
                  {deleteTarget.employee_id} - {deleteTarget.full_name || 'Không rõ tên'}
                </strong>
                ?
              </p>

              <div style={{
                marginTop: '0.9rem',
                padding: '0.7rem 0.85rem',
                borderRadius: 10,
                background: '#fffbeb',
                border: '1px solid #fde68a',
                color: '#92400e',
                fontSize: '0.77rem',
                lineHeight: 1.5
              }}>
                Lưu ý: nếu backend chưa có API xóa embedding/ảnh, hãy thêm route DELETE tương ứng. Frontend đã thử gọi các endpoint phổ biến trước khi xóa hồ sơ.
              </div>
            </div>

            <div style={{
              padding: '0.95rem 1.25rem',
              background: '#f8fafc',
              borderTop: '1px solid #f3f4f6',
              display: 'flex',
              justifyContent: 'flex-end',
              gap: '0.7rem'
            }}>
              <button
                className="btn btn-secondary"
                onClick={() => setDeleteTarget(null)}
                disabled={loading}
              >
                Hủy
              </button>

              <button
                className="btn"
                onClick={confirmDelete}
                disabled={loading}
                style={{
                  background: '#dc2626',
                  color: '#fff',
                  boxShadow: '0 8px 18px rgba(220,38,38,.20)'
                }}
              >
                {loading ? '⏳ Đang xóa...' : '🗑️ Xóa nhân viên'}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
