import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Employees from './pages/Employees';
import FaceRegistration from './pages/FaceRegistration';
import RealtimeAttendance from './pages/RealtimeAttendance';
import AttendanceLogs from './pages/AttendanceLogs';
import AISystem from './pages/AISystem';
import Reports from './pages/Reports';
import Cameras from './pages/Cameras';
import Settings from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/employees" element={<Employees />} />
            <Route path="/face-registration" element={<FaceRegistration />} />
            <Route path="/realtime" element={<RealtimeAttendance />} />
            <Route path="/logs" element={<AttendanceLogs />} />
            <Route path="/ai" element={<AISystem />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/cameras" element={<Cameras />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
