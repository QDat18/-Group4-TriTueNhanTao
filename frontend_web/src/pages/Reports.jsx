import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';
import { TrendingUp, Users, Clock, UserX, BarChart3 } from 'lucide-react';
import { getReportSummary, getReportByDepartment } from '../api';

const PERIODS = [
  { key:'day',   label:'Hôm nay',  icon:'📅' },
  { key:'week',  label:'Tuần này', icon:'📆' },
  { key:'month', label:'Tháng này',icon:'🗓️' },
];

const CustomBarTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:10, padding:'0.625rem 0.875rem', boxShadow:'0 8px 24px rgba(0,0,0,.10)', fontSize:'0.8rem' }}>
      <div style={{ color:'#9ca3af', marginBottom:4, fontSize:'0.7rem', fontWeight:600 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ display:'flex', alignItems:'center', gap:6, marginBottom:2 }}>
          <span style={{ width:8, height:8, borderRadius:'50%', background:p.fill, display:'inline-block' }} />
          <span style={{ color:'#374151', fontWeight:600 }}>{p.name}: </span>
          <span style={{ color:'#111827', fontWeight:800 }}>{p.value}</span>
        </div>
      ))}
    </div>
  );
};

export default function Reports() {
  const [period, setPeriod] = useState('month');
  const [summary, setSummary] = useState(null);
  const [deptData, setDeptData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([getReportSummary(period), getReportByDepartment()])
      .then(([s, d]) => { setSummary(s); setDeptData(d.data || []); setLoading(false); });
  }, [period]);

  const periodLabel = PERIODS.find(p => p.key === period)?.label || '';

  const summaryCards = summary ? [
    { label:'Tỷ Lệ Đi Làm',  value:`${summary.attendance_rate}%`, sub:`${summary.working_days} ngày làm việc`,       color:'#16a34a', bg:'#f0fdf4', icon:Users },
    { label:'Tỷ Lệ Đi Muộn', value:`${summary.late_rate}%`,       sub:'So với tổng số lượt điểm danh',               color:'#d97706', bg:'#fffbeb', icon:Clock },
    { label:'Tỷ Lệ Vắng Mặt',value:`${summary.absent_rate}%`,     sub:`${summary.total_employees} nhân viên tổng`,   color:'#dc2626', bg:'#fef2f2', icon:UserX },
  ] : [];

  return (
    <div className="animate-in">
      {/* Header */}
      <div style={{ display:'flex', alignItems:'flex-end', justifyContent:'space-between', marginBottom:'1.5rem' }}>
        <div>
          <h1 style={{ fontSize:'1.625rem', fontWeight:800, color:'#111827', letterSpacing:'-0.025em' }}>Báo cáo thống kê</h1>
          <p style={{ fontSize:'0.875rem', color:'#9ca3af', marginTop:'0.25rem' }}>Thống kê chấm công theo {periodLabel.toLowerCase()}</p>
        </div>
        {/* Period tabs */}
        <div style={{ display:'flex', gap:'0.375rem', background:'#f3f4f6', borderRadius:10, padding:'0.25rem' }}>
          {PERIODS.map(p => (
            <button key={p.key} onClick={() => setPeriod(p.key)} style={{
              padding:'0.425rem 0.875rem', borderRadius:8, border:'none', cursor:'pointer',
              fontSize:'0.8rem', fontWeight:600, fontFamily:'inherit',
              background: period===p.key ? '#fff' : 'transparent',
              color: period===p.key ? '#111827' : '#6b7280',
              boxShadow: period===p.key ? '0 1px 3px rgba(0,0,0,.1)' : 'none',
              transition:'all 150ms ease',
            }}>
              {p.icon} {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Summary cards */}
      {loading ? (
        <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:'1rem', marginBottom:'1.25rem' }}>
          {[...Array(3)].map((_,i) => (
            <div key={i} style={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:16, padding:'1.375rem' }}>
              <div className="skeleton" style={{ width:44, height:44, borderRadius:12, marginBottom:16 }} />
              <div className="skeleton" style={{ width:'45%', height:28, borderRadius:6, marginBottom:8 }} />
              <div className="skeleton" style={{ width:'65%', height:11, borderRadius:4 }} />
            </div>
          ))}
        </div>
      ) : (
        <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:'1rem', marginBottom:'1.25rem' }}>
          {summaryCards.map(s => (
            <div key={s.label} style={{
              background:'#fff', border:'1px solid #e5e7eb', borderRadius:16, padding:'1.375rem',
              position:'relative', overflow:'hidden',
              boxShadow:'0 1px 3px rgba(0,0,0,.06)',
              transition:'transform 180ms ease, box-shadow 180ms ease',
            }}
            onMouseEnter={e => { e.currentTarget.style.transform='translateY(-2px)'; e.currentTarget.style.boxShadow='0 6px 20px rgba(0,0,0,.09)'; }}
            onMouseLeave={e => { e.currentTarget.style.transform='translateY(0)'; e.currentTarget.style.boxShadow='0 1px 3px rgba(0,0,0,.06)'; }}
            >
              <div style={{ position:'absolute', left:0, top:0, bottom:0, width:4, background:s.color, borderRadius:'16px 0 0 16px' }} />
              <div style={{ paddingLeft:8 }}>
                <div style={{ width:44, height:44, borderRadius:12, background:s.bg, display:'flex', alignItems:'center', justifyContent:'center', marginBottom:'1rem' }}>
                  <s.icon size={22} color={s.color} />
                </div>
                <div style={{ fontSize:'2.25rem', fontWeight:800, color:s.color, letterSpacing:'-0.04em', lineHeight:1, marginBottom:'0.3rem' }}>{s.value}</div>
                <div style={{ fontSize:'0.8rem', fontWeight:600, color:'#374151' }}>{s.label}</div>
                <div style={{ fontSize:'0.7rem', color:'#9ca3af', marginTop:2 }}>{s.sub}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="grid-2" style={{ marginBottom:'1.25rem' }}>
        {/* Daily chart */}
        <div style={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:16, padding:'1.375rem', boxShadow:'0 1px 3px rgba(0,0,0,.06)' }}>
          <div style={{ display:'flex', alignItems:'center', gap:'0.5rem', marginBottom:'1.125rem' }}>
            <div style={{ width:32, height:32, borderRadius:8, background:'#eff6ff', display:'flex', alignItems:'center', justifyContent:'center' }}>
              <TrendingUp size={16} color="#2563eb" />
            </div>
            <div>
              <div style={{ fontSize:'0.875rem', fontWeight:700, color:'#111827' }}>Biểu Đồ Theo Ngày</div>
              <div style={{ fontSize:'0.7rem', color:'#9ca3af' }}>{periodLabel}</div>
            </div>
          </div>
          {loading ? (
            <div className="skeleton" style={{ height:240, borderRadius:8 }} />
          ) : summary?.daily_data?.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={summary.daily_data} margin={{ top:4, right:4, left:-20, bottom:0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
                <XAxis dataKey="date" tick={{ fill:'#9ca3af', fontSize:10 }} tickFormatter={v => v.slice(5)} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill:'#9ca3af', fontSize:10 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomBarTooltip />} />
                <Bar dataKey="present" name="Đi làm" radius={[6,6,0,0]}>
                  {summary.daily_data.map((_, i) => (
                    <Cell key={i} fill={`hsl(${220 + i * 2}, 80%, ${55 + (i % 3) * 5}%)`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ textAlign:'center', padding:'3rem', color:'#9ca3af' }}>
              <div style={{ fontSize:'2rem', marginBottom:'0.5rem', opacity:0.4 }}>📊</div>
              <p style={{ fontSize:'0.875rem' }}>Chưa có dữ liệu {periodLabel.toLowerCase()}</p>
            </div>
          )}
        </div>

        {/* Department chart */}
        <div style={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:16, padding:'1.375rem', boxShadow:'0 1px 3px rgba(0,0,0,.06)' }}>
          <div style={{ display:'flex', alignItems:'center', gap:'0.5rem', marginBottom:'1.125rem' }}>
            <div style={{ width:32, height:32, borderRadius:8, background:'#f0fdf4', display:'flex', alignItems:'center', justifyContent:'center' }}>
              <BarChart3 size={16} color="#16a34a" />
            </div>
            <div>
              <div style={{ fontSize:'0.875rem', fontWeight:700, color:'#111827' }}>Theo Phòng Ban</div>
              <div style={{ fontSize:'0.7rem', color:'#9ca3af' }}>Đi làm vs Vắng mặt hôm nay</div>
            </div>
          </div>
          {loading ? (
            <div className="skeleton" style={{ height:240, borderRadius:8 }} />
          ) : deptData.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={deptData} layout="vertical" margin={{ top:4, right:16, left:0, bottom:0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" horizontal={false} />
                <XAxis type="number" tick={{ fill:'#9ca3af', fontSize:10 }} axisLine={false} tickLine={false} />
                <YAxis dataKey="department" type="category" tick={{ fill:'#374151', fontSize:11, fontWeight:500 }} width={72} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomBarTooltip />} />
                <Bar dataKey="present" name="Đi làm" fill="#22c55e" radius={[0,4,4,0]} />
                <Bar dataKey="absent"  name="Vắng"   fill="#f87171" radius={[0,4,4,0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ textAlign:'center', padding:'3rem', color:'#9ca3af' }}>
              <div style={{ fontSize:'2rem', marginBottom:'0.5rem', opacity:0.4 }}>🏢</div>
              <p style={{ fontSize:'0.875rem' }}>Chưa có dữ liệu phòng ban</p>
            </div>
          )}
        </div>
      </div>

      {/* Dept detail table */}
      {!loading && deptData.length > 0 && (
        <div style={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:16, padding:'1.375rem', boxShadow:'0 1px 3px rgba(0,0,0,.06)' }}>
          <div style={{ display:'flex', alignItems:'center', gap:'0.5rem', marginBottom:'1.125rem' }}>
            <div style={{ width:32, height:32, borderRadius:8, background:'#faf5ff', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'1rem' }}>🏆</div>
            <div>
              <div style={{ fontSize:'0.875rem', fontWeight:700, color:'#111827' }}>Chi Tiết Theo Phòng Ban</div>
              <div style={{ fontSize:'0.7rem', color:'#9ca3af' }}>Xếp hạng chuyên cần hôm nay</div>
            </div>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Phòng ban</th>
                <th style={{ textAlign:'center' }}>Tổng NV</th>
                <th style={{ textAlign:'center' }}>Đi làm</th>
                <th style={{ textAlign:'center' }}>Vắng mặt</th>
                <th style={{ textAlign:'center' }}>Tỷ lệ</th>
                <th>Biểu đồ</th>
              </tr>
            </thead>
            <tbody>
              {[...deptData].sort((a,b) => b.rate - a.rate).map((dept, i) => {
                const rateColor = dept.rate>=90?'#15803d':dept.rate>=70?'#d97706':'#dc2626';
                const rateBg    = dept.rate>=90?'#f0fdf4':dept.rate>=70?'#fffbeb':'#fef2f2';
                return (
                  <tr key={dept.department}>
                    <td>
                      <div style={{ display:'flex', alignItems:'center', gap:'0.5rem' }}>
                        <span>{['🥇','🥈','🥉'][i]||`#${i+1}`}</span>
                        <span style={{ fontWeight:600 }}>{dept.department}</span>
                      </div>
                    </td>
                    <td style={{ textAlign:'center', fontWeight:600 }}>{dept.total}</td>
                    <td style={{ textAlign:'center', color:'#16a34a', fontWeight:700 }}>{dept.present}</td>
                    <td style={{ textAlign:'center', color:'#dc2626', fontWeight:700 }}>{dept.absent}</td>
                    <td style={{ textAlign:'center' }}>
                      <span style={{ fontSize:'0.75rem', fontWeight:700, padding:'0.2rem 0.625rem', borderRadius:99, background:rateBg, color:rateColor }}>
                        {dept.rate}%
                      </span>
                    </td>
                    <td style={{ width:140 }}>
                      <div style={{ height:6, background:'#f3f4f6', borderRadius:99, overflow:'hidden' }}>
                        <div style={{ height:'100%', borderRadius:99, width:`${dept.rate}%`, background:`linear-gradient(90deg,${rateColor},${rateColor}99)` }} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}