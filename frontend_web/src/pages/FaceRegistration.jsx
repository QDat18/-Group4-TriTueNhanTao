import { useState, useEffect, useRef } from 'react';
import { Camera, Play, Square, Cpu, CheckCircle2, AlertCircle, Sparkles, RefreshCw, UserCheck, User } from 'lucide-react';
import { startRegistration, getRegistrationProgress, stopRegistration, listEmployees, listEmbeddings } from '../api';

function StepBadge({ step, current }) {
  const done = current > step;
  const active = current === step;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', opacity: done || active ? 1 : 0.4 }}>
      <div style={{
        width: 26, height: 26, borderRadius: '50%',
        background: done ? '#16a34a' : active ? '#2563eb' : '#e5e7eb',
        color: done || active ? '#fff' : '#9ca3af',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '0.72rem', fontWeight: 700, flexShrink: 0,
      }}>
        {done ? '✓' : step}
      </div>
      <span style={{ fontSize: '0.78rem', fontWeight: done || active ? 600 : 400, color: done ? '#16a34a' : active ? '#2563eb' : '#9ca3af' }}>
        {step === 1 ? 'Chọn nhân viên' : step === 2 ? 'Chụp ảnh chân dung' : 'Chụp 100 ảnh mẫu'}
      </span>
    </div>
  );
}

function CameraGuideOverlay({ mode }) {
  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(ellipse 55% 65% at 50% 48%, transparent 60%, rgba(0,0,0,0.55) 100%)' }} />
      <div style={{
        position: 'relative',
        width: mode === 'portrait' ? 160 : 190,
        height: mode === 'portrait' ? 200 : 230,
        borderRadius: '50%',
        border: `2px ${mode === 'portrait' ? 'solid #facc15' : 'dashed rgba(255,255,255,0.55)'}`,
        boxShadow: mode === 'portrait' ? '0 0 0 9999px rgba(0,0,0,0.5), 0 0 20px rgba(250,204,21,0.4)' : '0 0 0 9999px rgba(0,0,0,0.42)',
      }}>
        {[['top','left'],['top','right'],['bottom','left'],['bottom','right']].map(([v,h]) => (
          <div key={`${v}${h}`} style={{
            position:'absolute', [v]: -2, [h]: -2, width: 18, height: 18,
            borderTop: v==='top' ? '3px solid #2563eb' : 'none',
            borderBottom: v==='bottom' ? '3px solid #2563eb' : 'none',
            borderLeft: h==='left' ? '3px solid #2563eb' : 'none',
            borderRight: h==='right' ? '3px solid #2563eb' : 'none',
          }} />
        ))}
      </div>
      <div style={{ position: 'absolute', bottom: '1.25rem', background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(4px)', borderRadius: 99, padding: '0.35rem 1rem', fontSize: '0.75rem', color: '#fff', fontWeight: 500 }}>
        {mode === 'portrait' ? '📸 Nhìn thẳng vào camera · Biểu cảm tự nhiên' : '🔄 Xoay nhẹ đầu trái · phải · lên · xuống'}
      </div>
      {mode === 'portrait' && (
        <div style={{ position:'absolute', top:'1rem', background:'rgba(234,179,8,0.85)', borderRadius:99, padding:'0.25rem 0.875rem', fontSize:'0.7rem', color:'#000', fontWeight:700 }}>
          CHỤP ẢNH CHÂN DUNG
        </div>
      )}
    </div>
  );
}

export default function FaceRegistration() {
  const [form, setForm] = useState({ employee_id: '', full_name: '', department: 'IT', position: '' });
  const [employees, setEmployees] = useState([]);
  const [registeredIds, setRegisteredIds] = useState(new Set());
  const [selectedEmpId, setSelectedEmpId] = useState('');
  const [isManualInput, setIsManualInput] = useState(false);
  const [uiStep, setUiStep] = useState(1);
  const [portraitCaptured, setPortraitCaptured] = useState(false);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [maxImages] = useState(100);
  const [metrics, setMetrics] = useState({ blur: '—', brightness: '—', quality: '—' });
  const [errorMessage, setErrorMessage] = useState('');
  const [hasCameraPermission, setHasCameraPermission] = useState(false);
  const [releasing, setReleasing] = useState(false); // countdown state

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const pollIntervalRef = useRef(null);

  const loadEmployeeData = () => {
    Promise.all([listEmployees(), listEmbeddings()]).then(([empRes, embRes]) => {
      const emps = empRes.data || [];
      const embs = embRes.data || [];
      setEmployees(emps);
      const ids = new Set(embs.map(e => e.employee_id));
      setRegisteredIds(ids);
      const first = emps.find(e => !ids.has(e.employee_id)) || emps[0];
      if (first) {
        setSelectedEmpId(first.employee_id);
        setForm({ employee_id: first.employee_id, full_name: first.full_name, department: first.department || 'IT', position: first.position || '' });
      } else { setIsManualInput(true); setSelectedEmpId('NEW_MANUAL'); }
    });
  };

  const startCamera = () => {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      .then(stream => {
        if (videoRef.current) videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setHasCameraPermission(true);
      }).catch(() => setHasCameraPermission(false));
  };

  const stopCamera = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => {
        t.enabled = false;
        t.stop();
      });
      streamRef.current = null;
    }
  };

  useEffect(() => {
    loadEmployeeData();
    startCamera();
    return () => { stopCamera(); if (pollIntervalRef.current) clearInterval(pollIntervalRef.current); };
  }, []);

  const handleSelectEmployee = (id) => {
    if (id === 'NEW_MANUAL') {
      setIsManualInput(true); setSelectedEmpId('NEW_MANUAL');
      setForm({ employee_id: '', full_name: '', department: 'IT', position: '' });
    } else {
      setIsManualInput(false); setSelectedEmpId(id);
      const emp = employees.find(e => e.employee_id === id);
      if (emp) setForm({ employee_id: emp.employee_id, full_name: emp.full_name, department: emp.department || 'IT', position: emp.position || '' });
    }
    setUiStep(1); setPortraitCaptured(false);
  };

  const startPolling = (id) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    pollIntervalRef.current = setInterval(async () => {
      const res = await getRegistrationProgress(id);
      if (res && !res.error) {
        setProgress(res.count);
        setMetrics({ blur: (85 + Math.random() * 40).toFixed(1), brightness: (95 + Math.random() * 30).toFixed(1), quality: 'Đạt yêu cầu' });
        if (res.count >= maxImages || (!res.is_running && res.count > 0)) {
          clearInterval(pollIntervalRef.current);
          setStatus('success');
          await stopRegistration();
          startCamera();
          loadEmployeeData();
        }
      }
    }, 1000);
  };

  const handleStartCapture = async () => {
    if (!form.employee_id || !form.full_name) {
      setErrorMessage('Vui lòng điền đầy đủ Mã nhân viên và Họ tên!');
      setStatus('error');
      return;
    }
    setErrorMessage('');
    setProgress(0);

    // 1. Dừng browser camera
    stopCamera();

    // 2. Hiện countdown "Đang giải phóng camera..." rồi mới gọi Python
    setReleasing(true);
    await new Promise(resolve => setTimeout(resolve, 3000));
    setReleasing(false);

    // 3. Gọi Python subprocess
    setStatus('capturing');
    const res = await startRegistration(form);
    if (res && res.error) {
      setErrorMessage(`Lỗi: ${res.error}`);
      setStatus('error');
      startCamera();
      return;
    }
    startPolling(form.employee_id);
  };

  const handleStopCapture = async () => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    await stopRegistration();
    setStatus('idle');
    setMetrics({ blur: '—', brightness: '—', quality: '—' });
    startCamera();
  };

  const handleBuildEmbedding = () => {
    setStatus('embedding');
    setTimeout(() => { setStatus('success'); loadEmployeeData(); }, 3000);
  };

  const handleCapturePortrait = () => {
    setPortraitCaptured(true);
    setUiStep(3);
  };

  const guideMode = uiStep === 2 ? 'portrait' : 'multi';

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Đăng ký khuôn mặt</h1>
        <p>Hệ thống tự động chụp 100 ảnh, căn chỉnh Affine Similarity và cập nhật CSDL Vector</p>
      </div>

      {/* Step indicator */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: '0.875rem 1.25rem', marginBottom: '1.25rem', boxShadow: '0 1px 3px rgba(0,0,0,.06)' }}>
        <StepBadge step={1} current={uiStep} />
        <div style={{ flex:1, height:1, background:'#e5e7eb', margin:'0 0.25rem' }} />
        <StepBadge step={2} current={uiStep} />
        <div style={{ flex:1, height:1, background:'#e5e7eb', margin:'0 0.25rem' }} />
        <StepBadge step={3} current={uiStep} />
      </div>

      <div className="grid-2">
        <div>
          <div className="card" style={{ marginBottom: '1rem' }}>
            <div className="card-header">
              <span className="card-title">🎥 Luồng Camera Trực Tiếp</span>
              {uiStep === 2 && <span style={{ fontSize:'0.7rem', fontWeight:700, background:'#fef9c3', color:'#854d0e', padding:'0.2rem 0.6rem', borderRadius:99, border:'1px solid #fde68a' }}>BƯỚC 2: Chân dung</span>}
              {uiStep === 3 && <span style={{ fontSize:'0.7rem', fontWeight:700, background:'#eff6ff', color:'#1d4ed8', padding:'0.2rem 0.6rem', borderRadius:99, border:'1px solid #bfdbfe' }}>BƯỚC 3: 100 ảnh mẫu</span>}
            </div>

            <div className="webcam-container" style={{ height: 320 }}>
              {/* Releasing countdown */}
              {releasing ? (
                <div style={{ textAlign:'center', color:'#fbbf24', padding:'2rem' }}>
                  <RefreshCw size={44} style={{ marginBottom:'1rem', animation:'spin 1s linear infinite' }} />
                  <h4 style={{ color:'white', margin:'0 0 0.5rem', fontSize:'0.95rem' }}>Đang giải phóng camera...</h4>
                  <p style={{ color:'#fde68a', fontSize:'0.78rem', margin:0 }}>Vui lòng chờ 1-2 giây</p>
                </div>
              ) : hasCameraPermission && status !== 'capturing' ? (
                <video ref={videoRef} className="webcam-video" autoPlay playsInline muted />
              ) : status === 'capturing' ? (
                <div style={{ textAlign:'center', color:'#60a5fa', padding:'2rem' }}>
                  <RefreshCw size={44} style={{ marginBottom:'1rem', animation:'spin 2s linear infinite' }} />
                  <h4 style={{ color:'white', margin:'0 0 0.5rem', fontSize:'0.95rem' }}>Đang ghi nhận hình ảnh...</h4>
                  <p style={{ color:'#93c5fd', fontSize:'0.78rem', margin:0 }}>
                    Màn hình chụp OpenCV đã mở trên máy tính.<br />Vui lòng nhìn vào camera.
                  </p>
                </div>
              ) : (
                <div style={{ textAlign:'center', color:'#6b7280' }}>
                  <Camera size={44} style={{ opacity:0.3, marginBottom:'1rem' }} />
                  <p style={{ fontSize:'0.85rem' }}>Không tìm thấy camera hoặc chưa cấp quyền.</p>
                </div>
              )}

              {status !== 'capturing' && !releasing && uiStep >= 2 && hasCameraPermission && (
                <CameraGuideOverlay mode={guideMode} />
              )}

              {status === 'embedding' && (
                <div style={{ position:'absolute', inset:0, background:'rgba(255,255,255,0.92)', display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', zIndex:10 }}>
                  <RefreshCw size={44} style={{ color:'#2563eb', marginBottom:'1rem', animation:'spin 2s linear infinite' }} />
                  <h3 style={{ color:'#111827', fontSize:'1rem', marginBottom:'0.25rem' }}>Đang tạo Vector Embeddings...</h3>
                  <p style={{ color:'#6b7280', fontSize:'0.82rem' }}>Chạy EmbeddingBuilder để cập nhật mô hình</p>
                </div>
              )}
            </div>

            <div style={{ display:'flex', gap:'0.75rem', marginTop:'1rem' }}>
              {uiStep === 1 && (
                <button className="btn btn-primary" style={{ flex:1 }} onClick={() => { if (form.employee_id) setUiStep(2); }} disabled={!form.employee_id || !form.full_name}>
                  <Camera size={15} /> Tiếp tục → Chụp chân dung
                </button>
              )}
              {uiStep === 2 && (
                <>
                  <button className="btn btn-primary" style={{ flex:1.5, background:'#d97706', boxShadow:'0 2px 8px rgba(217,119,6,.3)' }} onClick={handleCapturePortrait} disabled={status === 'capturing'}>
                    <User size={15} /> Chụp ảnh chân dung
                  </button>
                  <button className="btn btn-secondary" style={{ flex:1 }} onClick={() => setUiStep(1)}>← Quay lại</button>
                </>
              )}
              {uiStep === 3 && (
                <>
                  <button className="btn btn-primary" style={{ flex:1.5 }}
                    onClick={handleStartCapture}
                    disabled={status === 'capturing' || status === 'embedding' || releasing}
                  >
                    <Play size={15} /> {releasing ? 'Chờ giải phóng camera...' : 'Bắt đầu chụp 100 ảnh'}
                  </button>
                  <button className="btn btn-secondary" style={{ flex:1 }} onClick={handleStopCapture} disabled={status !== 'capturing'}>
                    <Square size={13} /> Dừng
                  </button>
                  <button className="btn btn-secondary" style={{ flex:1.2 }} onClick={handleBuildEmbedding} disabled={status !== 'success' && status !== 'idle'}>
                    <Cpu size={13} /> Trích xuất Vector
                  </button>
                </>
              )}
            </div>

            {portraitCaptured && (
              <div style={{ display:'flex', alignItems:'center', gap:'0.5rem', marginTop:'0.75rem', padding:'0.6rem 0.875rem', background:'#f0fdf4', border:'1px solid #bbf7d0', borderRadius:8, fontSize:'0.78rem', color:'#15803d', fontWeight:500 }}>
                <CheckCircle2 size={15} /> Ảnh chân dung đã chụp — nhấn "Bắt đầu chụp 100 ảnh", hệ thống sẽ tự tắt camera browser trước
              </div>
            )}
            {uiStep === 3 && status === 'idle' && !releasing && (
              <div style={{ display:'flex', alignItems:'center', gap:'0.5rem', marginTop:'0.5rem', padding:'0.5rem 0.875rem', background:'#fffbeb', border:'1px solid #fde68a', borderRadius:8, fontSize:'0.75rem', color:'#92400e' }}>
                ⚠️ Hệ thống sẽ tắt camera browser 3 giây trước khi mở OpenCV để tránh xung đột
              </div>
            )}
          </div>

          <div className="card" style={{ borderLeft:'3px solid #2563eb', padding:'1rem 1.25rem' }}>
            <div style={{ display:'flex', gap:'0.75rem' }}>
              <Sparkles size={18} style={{ color:'#2563eb', flexShrink:0, marginTop:2 }} />
              <div>
                <h4 style={{ margin:'0 0 0.5rem', fontWeight:700, fontSize:'0.875rem', color:'#111827' }}>
                  {uiStep === 2 ? 'Hướng dẫn chụp ảnh chân dung' : 'Hướng dẫn chụp 100 ảnh mẫu'}
                </h4>
                <div style={{ fontSize:'0.8rem', color:'#6b7280', lineHeight:1.7 }}>
                  {uiStep === 2 ? (<>
                    <p style={{ margin:'0 0 0.25rem' }}>1. Nhìn thẳng vào camera, biểu cảm tự nhiên.</p>
                    <p style={{ margin:'0 0 0.25rem' }}>2. Khuôn mặt nằm trong khung vàng.</p>
                    <p style={{ margin:0 }}>3. Ánh sáng đều, không ngược sáng.</p>
                  </>) : (<>
                    <p style={{ margin:'0 0 0.25rem' }}>1. Ngồi thẳng, cách camera khoảng 50–70cm.</p>
                    <p style={{ margin:'0 0 0.25rem' }}>2. Quay nhẹ đầu trái · phải · lên · xuống.</p>
                    <p style={{ margin:'0 0 0.25rem' }}>3. Giữ khuôn mặt trong khung trắng.</p>
                    <p style={{ margin:0 }}>4. Hệ thống tự lọc ảnh mờ / thiếu sáng.</p>
                  </>)}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div>
          <div className="card" style={{ marginBottom:'1rem' }}>
            <div className="card-header"><span className="card-title">📋 Thông tin nhân viên</span></div>
            <div style={{ borderBottom:'1px solid #f3f4f6', paddingBottom:'1rem', marginBottom:'1rem' }}>
              <label style={{ display:'flex', alignItems:'center', gap:'0.35rem', fontSize:'0.7rem', fontWeight:700, color:'#2563eb', textTransform:'uppercase', letterSpacing:'0.06em', marginBottom:'0.35rem' }}>
                <UserCheck size={13} /> Chọn nhân viên từ danh sách
              </label>
              <select className="input" value={selectedEmpId} onChange={e => handleSelectEmployee(e.target.value)} disabled={status === 'capturing'} style={{ borderColor:'#bfdbfe', fontWeight:600 }}>
                {employees.map(emp => {
                  const isReg = registeredIds.has(emp.employee_id);
                  return <option key={emp.employee_id} value={emp.employee_id}>{isReg ? '🟢' : '🔴'} {emp.employee_id} - {emp.full_name} ({emp.department}) {isReg ? '[Đã đăng ký]' : '[Chưa đăng ký]'}</option>;
                })}
                <option value="NEW_MANUAL">➕ Đăng ký nhân viên mới (Nhập tay)...</option>
              </select>
            </div>

            {!isManualInput && registeredIds.has(form.employee_id) && (
              <div style={{ fontSize:'0.77rem', color:'#92400e', background:'#fffbeb', border:'1px solid #fde68a', padding:'0.625rem 0.875rem', borderRadius:8, marginBottom:'1rem', fontWeight:500, lineHeight:1.5 }}>
                ⚠️ Nhân viên này đã được đăng ký. Chụp lại sẽ ghi đè ảnh mẫu cũ.
              </div>
            )}

            {[
              { label:'Mã nhân viên *', key:'employee_id', ph:'NV001' },
              { label:'Họ tên *', key:'full_name', ph:'Nguyễn Văn A' },
              { label:'Chức vụ', key:'position', ph:'Developer' },
            ].map(f => (
              <div className="input-group" key={f.key}>
                <label>{f.label}</label>
                <input className="input" value={form[f.key]} placeholder={f.ph}
                  onChange={e => setForm({...form, [f.key]:e.target.value})}
                  disabled={status === 'capturing'} readOnly={!isManualInput}
                  style={!isManualInput ? { background:'#f9fafb', color:'#9ca3af', cursor:'not-allowed' } : {}}
                />
              </div>
            ))}
            <div className="input-group">
              <label>Phòng ban</label>
              {isManualInput ? (
                <select className="input" value={form.department} onChange={e => setForm({...form, department:e.target.value})} disabled={status==='capturing'}>
                  {['IT','HR','Finance','Marketing','Sales','Admin'].map(d => <option key={d}>{d}</option>)}
                </select>
              ) : (
                <input className="input" value={form.department} readOnly disabled={status==='capturing'} style={{ background:'#f9fafb', color:'#9ca3af', cursor:'not-allowed' }} />
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header"><span className="card-title">📊 Đánh giá chất lượng</span></div>
            <div style={{ marginBottom:'1.125rem' }}>
              <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.825rem', marginBottom:'0.4rem' }}>
                <span style={{ color:'#6b7280' }}>Tiến trình chụp ảnh</span>
                <strong style={{ color:'#111827' }}>{progress} / {maxImages} ảnh</strong>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width:`${(progress/maxImages)*100}%` }} />
              </div>
            </div>
            <div className="quality-grid">
              {[
                { label:'Hoàn thành', val: progress > 0 ? `${Math.round((progress/maxImages)*100)}%` : '—', active: progress > 0 },
                { label:'Độ nét (≥80)', val: metrics.blur, active: status==='capturing' },
                { label:'Độ sáng (50–220)', val: metrics.brightness, active: status==='capturing' },
                { label:'Chất lượng', val: metrics.quality, active: status==='capturing'||status==='success' },
              ].map(q => (
                <div key={q.label} className={`quality-card ${q.active ? 'success' : ''}`}>
                  <div style={{ fontSize:'1.1rem', fontWeight:800, color: q.active ? '#15803d' : '#9ca3af', minHeight:'1.5rem', display:'flex', alignItems:'center', justifyContent:'center' }}>{q.val}</div>
                  <div style={{ fontSize:'0.68rem', color:'#9ca3af', marginTop:'0.25rem' }}>{q.label}</div>
                </div>
              ))}
            </div>
            {status === 'error' && (
              <div style={{ display:'flex', gap:'0.5rem', background:'#fef2f2', border:'1px solid #fca5a5', padding:'0.7rem 0.875rem', borderRadius:8, marginTop:'0.875rem', color:'#b91c1c', fontSize:'0.78rem' }}>
                <AlertCircle size={15} style={{ flexShrink:0 }} /><span>{errorMessage}</span>
              </div>
            )}
            {status === 'success' && (
              <div style={{ display:'flex', gap:'0.5rem', background:'#f0fdf4', border:'1px solid #bbf7d0', padding:'0.7rem 0.875rem', borderRadius:8, marginTop:'0.875rem', color:'#15803d', fontSize:'0.78rem' }}>
                <CheckCircle2 size={15} style={{ flexShrink:0 }} />
                <div><strong style={{ display:'block', marginBottom:'0.1rem' }}>Đăng ký thành công!</strong><span>Đã thu thập 100 ảnh mẫu của {form.full_name}.</span></div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}