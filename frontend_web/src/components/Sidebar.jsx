import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Users, Camera, Clock, FileText, Brain, BarChart3, Video, Settings } from 'lucide-react';

const navSections = [
  { title:'Tổng quan', links:[{ to:'/', icon:LayoutDashboard, label:'Bảng điều khiển' }]},
  { title:'Nhân sự', links:[
    { to:'/employees',         icon:Users,  label:'Quản lý nhân viên' },
    { to:'/face-registration', icon:Camera, label:'Đăng ký khuôn mặt' },
  ]},
  { title:'Chấm công', links:[
    { to:'/realtime', icon:Clock,    label:'Nhận diện trực tiếp' },
    { to:'/logs',     icon:FileText, label:'Nhật ký chấm công' },
  ]},
  { title:'Hệ thống AI', links:[{ to:'/ai', icon:Brain, label:'Dữ liệu Vector AI' }]},
  { title:'Cấu hình & Báo cáo', links:[
    { to:'/reports',  icon:BarChart3, label:'Báo cáo thống kê' },
    { to:'/cameras',  icon:Video,     label:'Quản lý Camera' },
    { to:'/settings', icon:Settings,  label:'Cấu hình hệ thống' },
  ]},
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">🧑‍💼</div>
        <div>
          <h1>HVNH FaceID</h1>
          <span>Hệ thống chấm công AI</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navSections.map((section) => (
          <div className="nav-section" key={section.title}>
            <div className="nav-section-title">{section.title}</div>
            {section.links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === '/'}
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                <link.icon className="nav-icon" size={16} />
                {link.label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <div style={{ padding:'0.875rem 1.125rem', borderTop:'1px solid rgba(255,255,255,0.05)', display:'flex', alignItems:'center', gap:'0.5rem' }}>
        <span style={{ width:6, height:6, borderRadius:'50%', background:'#22c55e', flexShrink:0, boxShadow:'0 0 5px rgba(34,197,94,.8)', display:'inline-block' }} />
        <span style={{ fontSize:'0.67rem', color:'#374151', opacity:0.4 }}>© 2026 Nhóm 4 · HVNH</span>
      </div>
    </aside>
  );
}
