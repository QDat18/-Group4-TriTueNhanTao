import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Users, Camera, Clock, FileText,
  Brain, BarChart3, Video, Settings
} from 'lucide-react';

const navSections = [
  {
    title: 'Overview',
    links: [
      { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
    ],
  },
  {
    title: 'People',
    links: [
      { to: '/employees', icon: Users, label: 'Employees' },
      { to: '/face-registration', icon: Camera, label: 'Face Registration' },
    ],
  },
  {
    title: 'Attendance',
    links: [
      { to: '/realtime', icon: Clock, label: 'Realtime' },
      { to: '/logs', icon: FileText, label: 'Attendance Logs' },
    ],
  },
  {
    title: 'AI System',
    links: [
      { to: '/ai', icon: Brain, label: 'AI & Embeddings' },
    ],
  },
  {
    title: 'Analytics',
    links: [
      { to: '/reports', icon: BarChart3, label: 'Reports' },
      { to: '/cameras', icon: Video, label: 'Cameras' },
      { to: '/settings', icon: Settings, label: 'Settings' },
    ],
  },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">🧑‍💼</div>
        <div>
          <h1>Face Attend</h1>
          <span>Management System</span>
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
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              >
                <link.icon className="nav-icon" size={18} />
                {link.label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <div style={{ padding: '1rem', borderTop: '1px solid var(--border-color)', textAlign: 'center' }}>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>© 2026 Group 4 - HVNH</span>
      </div>
    </aside>
  );
}
