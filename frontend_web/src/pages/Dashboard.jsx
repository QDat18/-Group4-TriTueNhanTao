import { useEffect, useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Users, UserCheck, Clock, UserX, TrendingUp, Award, Video } from 'lucide-react';
import { getDashboardStats, getAttendanceChart, getDepartmentRanking, listDevices } from '../api';

function SkeletonStatCards() {
  return (
    <div className="stats-grid">
      {[...Array(4)].map((_, i) => (
        <div key={i} style={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:16, padding:'1.5rem', boxShadow:'0 1px 3px rgba(0,0,0,.07)' }}>
          <div className="skeleton" style={{ width:44, height:44, borderRadius:12, marginBottom:18 }} />
          <div className="skeleton" style={{ width:'40%', height:32, borderRadius:6, marginBottom:10 }} />
          <div className="skeleton" style={{ width:'60%', height:12, borderRadius:4 }} />
        </div>
      ))}
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:10, padding:'0.65rem 1rem', boxShadow:'0 8px 24px rgba(0,0,0,.12)', fontSize:'0.8rem' }}>
      <div style={{ color:'#9ca3af', marginBottom:4, fontSize:'0.7rem', fontWeight:600, textTransform:'uppercase', letterSpacing:'0.05em' }}>{label}</div>
      <div style={{ fontWeight:800, color:'#1d4ed8', fontSize:'1.25rem', letterSpacing:'-0.02em' }}>
        {payload[0].value}
        <span style={{ fontWeight:500, fontSize:'0.75rem', color:'#9ca3af', marginLeft:4 }}>lượt chấm công</span>
      </div>
    </div>
  );
};

const CARD_CONFIGS = {
  blue:   { gradient:'linear-gradient(135deg,#2563eb,#3b82f6)', light:'#eff6ff', text:'#1d4ed8', icon:'#dbeafe' },
  green:  { gradient:'linear-gradient(135deg,#059669,#10b981)', light:'#f0fdf4', text:'#065f46', icon:'#d1fae5' },
  yellow: { gradient:'linear-gradient(135deg,#d97706,#f59e0b)', light:'#fffbeb', text:'#92400e', icon:'#fef3c7' },
  red:    { gradient:'linear-gradient(135deg,#dc2626,#ef4444)', light:'#fef2f2', text:'#991b1b', icon:'#fee2e2' },
};

export default function Dashboard() {
  const [stats,   setStats]   = useState(null);
  const [chart,   setChart]   = useState([]);
  const [ranking, setRanking] = useState([]);
  const [devices, setDevices] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getDashboardStats(), getAttendanceChart(30), getDepartmentRanking(), listDevices()])
      .then(([s,c,r,d]) => { setStats(s); setChart(c.data||[]); setRanking(r.data||[]); setDevices(d.data||[]); setLoading(false); });
  }, []);

  const statCards = stats ? [
    { label:'Tổng Nhân Viên',  sub:'Đã đăng ký hệ thống', value:stats.total_employees, icon:Users,     color:'blue'   },
    { label:'Đi Làm Hôm Nay', sub:'Đã chấm công',         value:stats.present_today,   icon:UserCheck, color:'green'  },
    { label:'Đi Muộn',        sub:'So với giờ quy định',   value:stats.late_today,       icon:Clock,     color:'yellow' },
    { label:'Vắng Mặt',       sub:'Chưa chấm công',        value:stats.absent_today,     icon:UserX,     color:'red'    },
  ] : [];

  const onlineCount = devices.filter(d => d.is_active).length;

  return (
    <div className="animate-in">
      {/* Header */}
      <div style={{ display:'flex', alignItems:'flex-end', justifyContent:'space-between', marginBottom:'1.75rem' }}>
        <div>
          <h1 style={{ fontSize:'1.625rem', fontWeight:800, color:'#111827', letterSpacing:'-0.025em', lineHeight:1.2 }}>Bảng điều khiển</h1>
          <p style={{ fontSize:'0.875rem', color:'#9ca3af', marginTop:'0.25rem' }}>Tổng quan hệ thống chấm công khuôn mặt</p>
        </div>
        <div style={{ fontSize:'0.75rem', color:'#9ca3af', background:'#f9fafb', border:'1px solid #e5e7eb', borderRadius:8, padding:'0.375rem 0.75rem', fontWeight:500 }}>
          📅 {new Date().toLocaleDateString('vi-VN', { weekday:'long', year:'numeric', month:'long', day:'numeric' })}
        </div>
      </div>

      {/* Stat Cards */}
      {loading ? <SkeletonStatCards /> : (
        <div className="stats-grid" style={{ marginBottom:'1.25rem' }}>
          {statCards.map((s) => {
            const cfg = CARD_CONFIGS[s.color];
            return (
              <div key={s.label} style={{
                background:'#fff',
                border:'1px solid #e5e7eb',
                borderRadius:16,
                padding:'1.375rem',
                position:'relative',
                overflow:'hidden',
                boxShadow:'0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04)',
                transition:'transform 180ms ease, box-shadow 180ms ease',
                cursor:'default',
              }}
              onMouseEnter={e => { e.currentTarget.style.transform='translateY(-3px)'; e.currentTarget.style.boxShadow='0 8px 24px rgba(0,0,0,.10)'; }}
              onMouseLeave={e => { e.currentTarget.style.transform='translateY(0)'; e.currentTarget.style.boxShadow='0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04)'; }}
              >
                {/* left accent bar */}
                <div style={{ position:'absolute', left:0, top:0, bottom:0, width:4, background:cfg.gradient, borderRadius:'16px 0 0 16px' }} />

                {/* top row */}
                <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between', marginBottom:'1rem', paddingLeft:8 }}>
                  <div style={{ width:44, height:44, borderRadius:12, background:cfg.icon, display:'flex', alignItems:'center', justifyContent:'center' }}>
                    <s.icon size={22} color={cfg.text} />
                  </div>
                  <div style={{ fontSize:'0.68rem', fontWeight:700, color:cfg.text, background:cfg.light, padding:'0.2rem 0.625rem', borderRadius:99, letterSpacing:'0.03em' }}>
                    Hôm nay
                  </div>
                </div>

                {/* value */}
                <div style={{ paddingLeft:8 }}>
                  <div style={{ fontSize:'2.5rem', fontWeight:800, color:cfg.text, lineHeight:1, letterSpacing:'-0.04em', marginBottom:'0.3rem' }}>
                    {s.value ?? '—'}
                  </div>
                  <div style={{ fontSize:'0.8rem', fontWeight:600, color:'#374151' }}>{s.label}</div>
                  <div style={{ fontSize:'0.7rem', color:'#9ca3af', marginTop:2 }}>{s.sub}</div>
                </div>

                {/* bg watermark removed */}
              </div>
            );
          })}
        </div>
      )}

      {/* Charts row */}
      <div className="grid-2" style={{ marginBottom:'1.25rem' }}>
        {/* Area chart */}
        <div style={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:16, padding:'1.375rem', boxShadow:'0 1px 3px rgba(0,0,0,.06)' }}>
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'1.125rem' }}>
            <div style={{ display:'flex', alignItems:'center', gap:'0.5rem' }}>
              <div style={{ width:32, height:32, borderRadius:8, background:'#eff6ff', display:'flex', alignItems:'center', justifyContent:'center' }}>
                <TrendingUp size={16} color="#2563eb" />
              </div>
              <div>
                <div style={{ fontSize:'0.875rem', fontWeight:700, color:'#111827' }}>Biểu Đồ Chấm Công</div>
                <div style={{ fontSize:'0.7rem', color:'#9ca3af' }}>30 ngày gần nhất</div>
              </div>
            </div>
            <div style={{ fontSize:'0.7rem', color:'#6b7280', background:'#f9fafb', border:'1px solid #e5e7eb', borderRadius:6, padding:'0.25rem 0.625rem', fontWeight:500 }}>
              Tháng này
            </div>
          </div>
          {loading ? (
            <div className="skeleton" style={{ height:220, borderRadius:8 }} />
          ) : chart.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={chart} margin={{ top:4, right:4, left:-20, bottom:0 }}>
                <defs>
                  <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#2563eb" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" tick={{ fill:'#9ca3af', fontSize:10 }} tickFormatter={v => v.slice(5)} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill:'#9ca3af', fontSize:10 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="count" stroke="#2563eb" strokeWidth={2.5} fillOpacity={1} fill="url(#grad)" dot={false} activeDot={{ r:5, fill:'#2563eb', stroke:'#fff', strokeWidth:2.5 }} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ textAlign:'center', padding:'3rem', color:'#9ca3af' }}>
              <div style={{ fontSize:'2rem', marginBottom:'0.5rem', opacity:0.4 }}>📊</div>
              <p style={{ fontSize:'0.875rem' }}>Chưa có dữ liệu</p>
            </div>
          )}
        </div>

        {/* Ranking */}
        <div style={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:16, padding:'1.375rem', boxShadow:'0 1px 3px rgba(0,0,0,.06)' }}>
          <div style={{ display:'flex', alignItems:'center', gap:'0.5rem', marginBottom:'1.125rem' }}>
            <div style={{ width:32, height:32, borderRadius:8, background:'#fffbeb', display:'flex', alignItems:'center', justifyContent:'center' }}>
              <Award size={16} color="#d97706" />
            </div>
            <div>
              <div style={{ fontSize:'0.875rem', fontWeight:700, color:'#111827' }}>Top Phòng Ban Đúng Giờ</div>
              <div style={{ fontSize:'0.7rem', color:'#9ca3af' }}>Xếp hạng chuyên cần</div>
            </div>
          </div>

          {loading ? (
            <div>{[...Array(4)].map((_,i) => (
              <div key={i} style={{ marginBottom:'1rem' }}>
                <div style={{ display:'flex', justifyContent:'space-between', marginBottom:8 }}>
                  <div className="skeleton" style={{ width:'55%', height:12, borderRadius:4 }} />
                  <div className="skeleton" style={{ width:'18%', height:12, borderRadius:4 }} />
                </div>
                <div className="skeleton" style={{ height:6, borderRadius:99 }} />
              </div>
            ))}</div>
          ) : ranking.length > 0 ? ranking.slice(0,5).map((dept, i) => {
            const medals = ['🥇','🥈','🥉'];
            const rateColor = dept.rate>=90?'#059669':dept.rate>=70?'#d97706':'#dc2626';
            const rateBg    = dept.rate>=90?'#f0fdf4':dept.rate>=70?'#fffbeb':'#fef2f2';
            return (
              <div key={dept.department} style={{ marginBottom:'1rem' }}>
                <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'0.375rem' }}>
                  <div style={{ display:'flex', alignItems:'center', gap:'0.5rem' }}>
                    <span style={{ fontSize:'1rem' }}>{medals[i]||`#${i+1}`}</span>
                    <span style={{ fontSize:'0.825rem', fontWeight:600, color:'#374151' }}>{dept.department}</span>
                  </div>
                  <span style={{ fontSize:'0.72rem', fontWeight:700, padding:'0.15rem 0.55rem', borderRadius:99, background:rateBg, color:rateColor }}>
                    {dept.rate}%
                  </span>
                </div>
                <div style={{ height:6, background:'#f3f4f6', borderRadius:99, overflow:'hidden' }}>
                  <div style={{ height:'100%', borderRadius:99, width:`${dept.rate}%`, background:`linear-gradient(90deg, ${rateColor}, ${rateColor}88)`, transition:'width .5s ease' }} />
                </div>
              </div>
            );
          }) : <div style={{ textAlign:'center', padding:'2rem', color:'#9ca3af', fontSize:'0.875rem' }}>Chưa có dữ liệu</div>}
        </div>
      </div>

      {/* Camera */}
      <div style={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:16, padding:'1.375rem', boxShadow:'0 1px 3px rgba(0,0,0,.06)' }}>
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'1.125rem' }}>
          <div style={{ display:'flex', alignItems:'center', gap:'0.5rem' }}>
            <div style={{ width:32, height:32, borderRadius:8, background:'#f0fdf4', display:'flex', alignItems:'center', justifyContent:'center' }}>
              <Video size={16} color="#059669" />
            </div>
            <div>
              <div style={{ fontSize:'0.875rem', fontWeight:700, color:'#111827' }}>Camera Đang Hoạt Động</div>
              <div style={{ fontSize:'0.7rem', color:'#9ca3af' }}>Giám sát thời gian thực</div>
            </div>
          </div>
          {!loading && devices.length > 0 && (
            <div style={{ display:'flex', alignItems:'center', gap:6, fontSize:'0.75rem', color:'#059669', fontWeight:600, background:'#f0fdf4', border:'1px solid #bbf7d0', padding:'0.25rem 0.75rem', borderRadius:99 }}>
              <span style={{ width:7, height:7, borderRadius:'50%', background:'#16a34a', display:'inline-block', boxShadow:'0 0 5px rgba(22,163,74,.7)' }} />
              {onlineCount}/{devices.length} online
            </div>
          )}
        </div>

        {loading ? (
          <div className="grid-3">{[...Array(3)].map((_,i) => (
            <div key={i} style={{ padding:'1rem', border:'1px solid #e5e7eb', borderRadius:12 }}>
              <div className="skeleton" style={{ width:'48%', height:13, borderRadius:4, marginBottom:8 }} />
              <div className="skeleton" style={{ width:'75%', height:11, borderRadius:4, marginBottom:12 }} />
              <div className="skeleton" style={{ width:62, height:20, borderRadius:99 }} />
            </div>
          ))}</div>
        ) : devices.length > 0 ? (
          <div className="grid-3">
            {devices.map(dev => (
              <div key={dev.device_id} style={{
                padding:'1rem 1.125rem', borderRadius:12,
                border:`1px solid ${dev.is_active?'#bbf7d0':'#e5e7eb'}`,
                background:dev.is_active?'#f0fdf4':'#f9fafb',
                transition:'all 180ms ease',
              }}>
                <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'0.375rem' }}>
                  <div style={{ fontWeight:700, fontSize:'0.875rem', color:'#111827' }}>{dev.device_id}</div>
                  <span className={`badge ${dev.is_active?'badge-success':'badge-danger'}`}>{dev.is_active?'● Online':'○ Offline'}</span>
                </div>
                <div style={{ fontSize:'0.72rem', color:'#6b7280', lineHeight:1.5 }}>{dev.device_name}</div>
                <div style={{ fontSize:'0.7rem', color:'#9ca3af' }}>📍 {dev.location}</div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ textAlign:'center', padding:'2.5rem', color:'#9ca3af' }}>
            <div style={{ fontSize:'2.5rem', marginBottom:'0.5rem', opacity:0.35 }}>🎥</div>
            <p style={{ fontSize:'0.875rem' }}>Chưa có camera nào được thêm</p>
          </div>
        )}
      </div>
    </div>
  );
}
