import { useEffect, useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Users, UserCheck, Clock, UserX } from 'lucide-react';
import { getDashboardStats, getAttendanceChart, getDepartmentRanking, listDevices } from '../api';

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [chart, setChart] = useState([]);
  const [ranking, setRanking] = useState([]);
  const [devices, setDevices] = useState([]);

  useEffect(() => {
    getDashboardStats().then(setStats);
    getAttendanceChart(30).then(r => setChart(r.data || []));
    getDepartmentRanking().then(r => setRanking(r.data || []));
    listDevices().then(r => setDevices(r.data || []));
  }, []);

  if (!stats) return <div className="pulse" style={{ textAlign: 'center', padding: '4rem' }}>Loading...</div>;

  const statCards = [
    { label: 'Tổng Nhân Viên', value: stats.total_employees, icon: Users, color: 'blue' },
    { label: 'Đi Làm Hôm Nay', value: stats.present_today, icon: UserCheck, color: 'green' },
    { label: 'Đi Muộn', value: stats.late_today, icon: Clock, color: 'yellow' },
    { label: 'Vắng Mặt', value: stats.absent_today, icon: UserX, color: 'red' },
  ];

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Bảng điều khiển</h1>
        <p>Tổng quan hệ thống chấm công khuôn mặt</p>
      </div>

      {/* Stat Cards */}
      <div className="stats-grid">
        {statCards.map((s) => (
          <div className={`stat-card ${s.color}`} key={s.label}>
            <div className="stat-icon"><s.icon size={24} /></div>
            <div className="stat-value">{s.value}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">📈 Biểu Đồ Chấm Công 30 Ngày</span>
          </div>
          {chart.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chart}>
                <defs>
                  <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={v => v.slice(5)} />
                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: 8, color: 'var(--text-primary)' }} />
                <Area type="monotone" dataKey="count" stroke="var(--accent-primary)" fillOpacity={1} fill="url(#colorCount)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-state"><div className="empty-state-icon">📊</div><p>Chưa có dữ liệu</p></div>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">🏆 Top Phòng Ban Đúng Giờ</span>
          </div>
          {ranking.length > 0 ? ranking.slice(0, 5).map((dept, i) => (
            <div key={dept.department} style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.3rem' }}>
                <span>#{i + 1} {dept.department}</span>
                <span style={{ color: 'var(--accent-success)' }}>{dept.rate}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${dept.rate}%` }} />
              </div>
            </div>
          )) : (
            <div className="empty-state"><p>Chưa có dữ liệu</p></div>
          )}
        </div>
      </div>

      {/* Camera Status */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">🎥 Camera Đang Hoạt Động</span>
        </div>
        {devices.length > 0 ? (
          <div className="grid-3">
            {devices.map(dev => (
              <div key={dev.device_id} style={{ padding: '1rem', background: 'rgba(0,0,0,0.01)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontWeight: 600, marginBottom: '0.3rem' }}>{dev.device_id}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>{dev.device_name} - {dev.location}</div>
                <span className={`badge ${dev.is_active ? 'badge-success' : 'badge-danger'}`}>
                  {dev.is_active ? '● Online' : '● Offline'}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state"><div className="empty-state-icon">🎥</div><p>Chưa có camera</p></div>
        )}
      </div>
    </div>
  );
}
