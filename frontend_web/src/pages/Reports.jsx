import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { getReportSummary, getReportByDepartment } from '../api';

export default function Reports() {
  const [period, setPeriod] = useState('month');
  const [summary, setSummary] = useState(null);
  const [deptData, setDeptData] = useState([]);

  useEffect(() => {
    getReportSummary(period).then(setSummary);
    getReportByDepartment().then(r => setDeptData(r.data || []));
  }, [period]);

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Reports</h1>
        <p>Báo cáo chấm công</p>
      </div>

      {/* Period Selector */}
      <div className="tabs" style={{ marginBottom: '2rem' }}>
        {['day', 'week', 'month'].map(p => (
          <button key={p} className={`tab ${period === p ? 'active' : ''}`} onClick={() => setPeriod(p)}>
            {p === 'day' ? '📅 Ngày' : p === 'week' ? '📆 Tuần' : '🗓️ Tháng'}
          </button>
        ))}
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
          <div className="stat-card green"><div className="stat-value">{summary.attendance_rate}%</div><div className="stat-label">Tỷ Lệ Đi Làm</div></div>
          <div className="stat-card yellow"><div className="stat-value">{summary.late_rate}%</div><div className="stat-label">Tỷ Lệ Đi Muộn</div></div>
          <div className="stat-card red"><div className="stat-value">{summary.absent_rate}%</div><div className="stat-label">Tỷ Lệ Vắng Mặt</div></div>
        </div>
      )}

      <div className="grid-2">
        {/* Daily Chart */}
        <div className="card">
          <div className="card-header"><span className="card-title">📊 Biểu Đồ Theo Ngày</span></div>
          {summary?.daily_data?.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={summary.daily_data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} tickFormatter={v => v.slice(5)} />
                <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: 8, color: 'var(--text-primary)' }} />
                <Bar dataKey="present" fill="var(--accent-primary)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div className="empty-state"><p>Chưa có dữ liệu</p></div>}
        </div>

        {/* Department Chart */}
        <div className="card">
          <div className="card-header"><span className="card-title">🏢 Theo Phòng Ban</span></div>
          {deptData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={deptData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                <YAxis dataKey="department" type="category" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} width={80} />
                <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: 8, color: 'var(--text-primary)' }} />
                <Bar dataKey="present" fill="var(--accent-success)" radius={[0, 4, 4, 0]} name="Đi làm" />
                <Bar dataKey="absent" fill="var(--accent-danger)" radius={[0, 4, 4, 0]} name="Vắng" />
              </BarChart>
            </ResponsiveContainer>
          ) : <div className="empty-state"><p>Chưa có dữ liệu</p></div>}
        </div>
      </div>
    </div>
  );
}
