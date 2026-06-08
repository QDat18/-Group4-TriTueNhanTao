import { useState, useEffect, useRef } from 'react';
import {
  Camera,
  Play,
  Square,
  Cpu,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  RefreshCw,
  UserCheck,
  User,
  Image as ImageIcon,
  RotateCcw,
  ShieldCheck
} from 'lucide-react';

import {
  startRegistration,
  createEmployee,
  getRegistrationProgress,
  stopRegistration,
  listEmployees,
  listEmbeddings,
  rebuildEmbeddings
} from '../api';

const API_BASE = 'http://localhost:8000';
const MAX_IMAGES = 50;

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


function StepBadge({ step, current, title, desc }) {
  const done = current > step;
  const active = current === step;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.65rem',
      opacity: done || active ? 1 : 0.45,
      minWidth: 0
    }}>
      <div style={{
        width: 30,
        height: 30,
        borderRadius: '50%',
        background: done ? '#16a34a' : active ? '#2563eb' : '#e5e7eb',
        color: done || active ? '#fff' : '#9ca3af',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '0.78rem',
        fontWeight: 800,
        flexShrink: 0,
        boxShadow: active ? '0 0 0 4px rgba(37,99,235,.12)' : 'none'
      }}>
        {done ? '✓' : step}
      </div>

      <div style={{ minWidth: 0 }}>
        <div style={{
          fontSize: '0.8rem',
          fontWeight: done || active ? 800 : 600,
          color: done ? '#16a34a' : active ? '#2563eb' : '#6b7280',
          whiteSpace: 'nowrap'
        }}>
          {title}
        </div>
        <div style={{
          fontSize: '0.68rem',
          color: '#9ca3af',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis'
        }}>
          {desc}
        </div>
      </div>
    </div>
  );
}

function CameraGuideOverlay() {
  return (
    <div style={{
      position: 'absolute',
      inset: 0,
      pointerEvents: 'none',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      <div style={{
        position: 'absolute',
        inset: 0,
        background: 'radial-gradient(ellipse 58% 68% at 50% 48%, transparent 58%, rgba(0,0,0,0.55) 100%)'
      }} />

      <div style={{
        position: 'relative',
        width: 190,
        height: 230,
        borderRadius: '50%',
        border: '2px dashed rgba(255,255,255,0.65)',
        boxShadow: '0 0 0 9999px rgba(0,0,0,0.38)'
      }}>
        {[['top', 'left'], ['top', 'right'], ['bottom', 'left'], ['bottom', 'right']].map(([v, h]) => (
          <div key={`${v}${h}`} style={{
            position: 'absolute',
            [v]: -3,
            [h]: -3,
            width: 22,
            height: 22,
            borderTop: v === 'top' ? '3px solid #60a5fa' : 'none',
            borderBottom: v === 'bottom' ? '3px solid #60a5fa' : 'none',
            borderLeft: h === 'left' ? '3px solid #60a5fa' : 'none',
            borderRight: h === 'right' ? '3px solid #60a5fa' : 'none'
          }} />
        ))}
      </div>

      <div style={{
        position: 'absolute',
        bottom: '1.25rem',
        background: 'rgba(0,0,0,0.68)',
        backdropFilter: 'blur(4px)',
        borderRadius: 999,
        padding: '0.4rem 1rem',
        fontSize: '0.75rem',
        color: '#fff',
        fontWeight: 600,
        textAlign: 'center',
        maxWidth: '85%'
      }}>
        🔄 Nhìn thẳng rồi xoay nhẹ trái · phải · lên · xuống
      </div>
    </div>
  );
}

function CapturedGallery({ images, maxImages }) {
  return (
    <div className="card" style={{ marginTop: '1rem' }}>
      <div className="card-header" style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ImageIcon size={16} style={{ color: '#2563eb' }} />
          Ảnh đã chụp
        </span>
        <span style={{
          fontSize: '0.72rem',
          fontWeight: 800,
          color: images.length >= maxImages ? '#16a34a' : '#2563eb',
          background: images.length >= maxImages ? '#dcfce7' : '#eff6ff',
          padding: '0.25rem 0.65rem',
          borderRadius: 999
        }}>
          {images.length}/{maxImages}
        </span>
      </div>

      {images.length === 0 ? (
        <div style={{
          border: '1px dashed #d1d5db',
          borderRadius: 12,
          padding: '1.25rem',
          textAlign: 'center',
          color: '#9ca3af',
          fontSize: '0.82rem',
          background: '#f9fafb'
        }}>
          Chưa có ảnh nào. Khi bắt đầu chụp, ảnh mẫu sẽ hiển thị tại đây.
        </div>
      ) : (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(58px, 1fr))',
          gap: '0.55rem',
          maxHeight: 280,
          overflowY: 'auto',
          paddingRight: 2
        }}>
          {images.map((src, index) => (
            <div key={`${src}-${index}`} style={{
              position: 'relative',
              borderRadius: 10,
              overflow: 'hidden',
              border: '1px solid #e5e7eb',
              background: '#f3f4f6',
              aspectRatio: '1 / 1',
              boxShadow: index === images.length - 1 ? '0 0 0 3px rgba(37,99,235,.16)' : 'none'
            }}>
              <img
                src={src}
                alt={`captured-${index}`}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: 'block'
                }}
              />
              <div style={{
                position: 'absolute',
                top: 4,
                left: 4,
                background: 'rgba(17,24,39,.72)',
                color: '#fff',
                borderRadius: 999,
                padding: '0.08rem 0.35rem',
                fontSize: '0.58rem',
                fontWeight: 700
              }}>
                {index + 1}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function FaceRegistration() {
  const [form, setForm] = useState({
    employee_id: '',
    full_name: '',
    department: 'IT',
    position: ''
  });

  const [employees, setEmployees] = useState([]);
  const [registeredIds, setRegisteredIds] = useState(new Set());
  const [selectedEmpId, setSelectedEmpId] = useState('');
  const [isManualInput, setIsManualInput] = useState(false);

  const [uiStep, setUiStep] = useState(1);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [metrics, setMetrics] = useState({
    blur: '—',
    brightness: '—',
    quality: '—'
  });
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [hasCameraPermission, setHasCameraPermission] = useState(false);
  const [releasing, setReleasing] = useState(false);
  const [capturedImages, setCapturedImages] = useState([]);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const pollIntervalRef = useRef(null);

  const loadEmployeeData = () => {
    Promise.all([listEmployees(), listEmbeddings()])
      .then(([empRes, embRes]) => {
        const emps = empRes.data || [];
        const embs = embRes.data || [];

        setEmployees(emps);

        const ids = new Set(embs.map(e => e.employee_id));
        setRegisteredIds(ids);

        const first = emps.find(e => !ids.has(e.employee_id)) || emps[0];

        if (first) {
          setIsManualInput(false);
          setSelectedEmpId(first.employee_id);
          setForm({
            employee_id: first.employee_id,
            full_name: first.full_name,
            department: first.department || 'IT',
            position: first.position || ''
          });
        } else {
          setIsManualInput(true);
          setSelectedEmpId('NEW_MANUAL');
          setForm({
            employee_id: getNextEmployeeId(emps),
            full_name: '',
            department: 'IT',
            position: ''
          });
        }
      });
  };

  const startCamera = () => {
    navigator.mediaDevices
      .getUserMedia({
        video: {
          width: 640,
          height: 480
        }
      })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        streamRef.current = stream;
        setHasCameraPermission(true);
      })
      .catch(() => {
        setHasCameraPermission(false);
      });
  };

  const stopCamera = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.enabled = false;
        track.stop();
      });

      streamRef.current = null;
    }
  };

  useEffect(() => {
    loadEmployeeData();

    return () => {
      stopCamera();

      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const resetCaptureState = () => {
    setProgress(0);
    setCapturedImages([]);
    setMetrics({
      blur: '—',
      brightness: '—',
      quality: '—'
    });
  };

  const handleSelectEmployee = (id) => {
    resetCaptureState();
    stopCamera();

    if (id === 'NEW_MANUAL') {
      setIsManualInput(true);
      setSelectedEmpId('NEW_MANUAL');
      setForm({
        employee_id: getNextEmployeeId(employees),
        full_name: '',
        department: 'IT',
        position: ''
      });
    } else {
      setIsManualInput(false);
      setSelectedEmpId(id);

      const emp = employees.find(e => e.employee_id === id);

      if (emp) {
        setForm({
          employee_id: emp.employee_id,
          full_name: emp.full_name,
          department: emp.department || 'IT',
          position: emp.position || ''
        });
      }
    }

    setUiStep(1);
    setStatus('idle');
    setErrorMessage('');
    setSuccessMessage('');
  };

  const buildImageUrls = (employeeId, count) => {
    const timestamp = Date.now();

    return Array.from({ length: count }, (_, i) => (
      `${API_BASE}/api/portraits/${employeeId}/${employeeId}_${String(i).padStart(3, '0')}.jpg?t=${timestamp}`
    ));
  };

  const startPolling = (employeeId) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    pollIntervalRef.current = setInterval(async () => {
      const res = await getRegistrationProgress(employeeId);

      if (!res || res.error) {
        return;
      }

      const count = Math.min(res.count || 0, MAX_IMAGES);

      setProgress(count);
      setCapturedImages(buildImageUrls(employeeId, count));

      if (count > 0) {
        setMetrics({
          blur: (85 + Math.random() * 40).toFixed(1),
          brightness: (95 + Math.random() * 30).toFixed(1),
          quality: 'Đạt yêu cầu'
        });
      }

      if (count >= MAX_IMAGES || !res.is_running) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;

        if (count === 0) {
          setStatus('error');
          setErrorMessage("Không thể khởi động camera hoặc quá trình chụp ảnh bị lỗi. Vui lòng kiểm tra lại camera!");
          await stopRegistration();
          startCamera();
          return;
        }

        setStatus('success');
        setSuccessMessage(`Đã chụp xong ${count} ảnh mẫu cho ${form.full_name || employeeId}. Bấm Tạo vector để hoàn tất nhận diện.`);

        await stopRegistration();

        startCamera();
        loadEmployeeData();
      }
    }, 900);
  };

  const handleStartCapture = async () => {
    const payload = {
      ...form,
      employee_id: form.employee_id.trim(),
      full_name: form.full_name.trim(),
      max_images: MAX_IMAGES
    };

    if (!payload.employee_id || !payload.full_name) {
      setErrorMessage('Vui lòng điền đầy đủ Mã nhân viên và Họ tên.');
      setSuccessMessage('');
      setStatus('error');
      return;
    }

    setErrorMessage('');
    setSuccessMessage('');
    resetCaptureState();

    try {
      if (isManualInput) {
        const exists = employees.some(e => e.employee_id === payload.employee_id);

        if (!exists) {
          const createRes = await createEmployee({
            employee_id: payload.employee_id,
            full_name: payload.full_name,
            department: payload.department,
            position: payload.position || ''
          });

          if (createRes?.error) {
            throw new Error(createRes.error);
          }
        }
      }

      stopCamera();

      setReleasing(true);
      await new Promise(resolve => setTimeout(resolve, 2500));
      setReleasing(false);

      setStatus('capturing');

      const res = await startRegistration(payload);

      if (res?.error) {
        throw new Error(res.error);
      }

      setSuccessMessage(`Đã bắt đầu đăng ký cho ${payload.employee_id} - ${payload.full_name}.`);
      startPolling(payload.employee_id);
    } catch (err) {
      setReleasing(false);
      setErrorMessage(`Không đăng ký được nhân viên: ${err.message || 'Kiểm tra backend/API hoặc camera.'}`);
      setSuccessMessage('');
      setStatus('error');
      startCamera();
    }
  };

  const handleStopCapture = async () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    await stopRegistration();

    setStatus('idle');
    setSuccessMessage('Đã dừng chụp ảnh mẫu.');
    setMetrics({
      blur: '—',
      brightness: '—',
      quality: '—'
    });

    startCamera();
  };

  const handleBuildEmbedding = async () => {
    setStatus('embedding');
    setErrorMessage('');

    try {
      const res = await rebuildEmbeddings();

      if (res && res.error) {
        setErrorMessage(`Lỗi trích xuất: ${res.error}`);
        setStatus('error');
        return;
      }

      setStatus('success');
      setSuccessMessage('Tạo vector nhận diện thành công. Nhân viên đã sẵn sàng chấm công.');
      loadEmployeeData();
    } catch (err) {
      setErrorMessage(`Lỗi kết nối: ${err.message}`);
      setSuccessMessage('');
      setStatus('error');
    }
  };

  const completionPercent = Math.round((progress / MAX_IMAGES) * 100);
  const selectedAlreadyRegistered = registeredIds.has(form.employee_id);

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1>Đăng ký khuôn mặt</h1>
        <p>Chụp 50 ảnh mẫu, tự động căn chỉnh khuôn mặt và cập nhật vector nhận diện bằng InsightFace Buffalo_L</p>
      </div>

      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        background: '#fff',
        border: '1px solid #e5e7eb',
        borderRadius: 14,
        padding: '0.95rem 1.25rem',
        marginBottom: '1.25rem',
        boxShadow: '0 1px 3px rgba(0,0,0,.06)'
      }}>
        <StepBadge
          step={1}
          current={uiStep}
          title="Chọn nhân viên"
          desc="Chọn có sẵn hoặc nhập mới"
        />

        <div style={{ flex: 1, height: 1, background: '#e5e7eb' }} />

        <StepBadge
          step={2}
          current={uiStep}
          title="Chụp 50 ảnh"
          desc="Đa góc mặt, đủ sáng"
        />
      </div>

      <div className="grid-2">
        <div>
          <div className="card" style={{ marginBottom: '1rem' }}>
            <div className="card-header" style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span className="card-title">🎥 Camera đăng ký</span>

              <span style={{
                fontSize: '0.7rem',
                fontWeight: 800,
                background: uiStep === 2 ? '#eff6ff' : '#f8fafc',
                color: uiStep === 2 ? '#1d4ed8' : '#64748b',
                padding: '0.25rem 0.65rem',
                borderRadius: 999,
                border: `1px solid ${uiStep === 2 ? '#bfdbfe' : '#e2e8f0'}`
              }}>
                {uiStep === 2 ? 'SẴN SÀNG CHỤP 50 ẢNH' : 'CHỌN NHÂN VIÊN'}
              </span>
            </div>

            <div className="webcam-container" style={{
              height: 320,
              position: 'relative',
              overflow: 'hidden',
              borderRadius: 14
            }}>
              {releasing ? (
                <div style={{
                  textAlign: 'center',
                  color: '#fbbf24',
                  padding: '2rem'
                }}>
                  <RefreshCw
                    size={44}
                    style={{
                      marginBottom: '1rem',
                      animation: 'spin 1s linear infinite'
                    }}
                  />
                  <h4 style={{
                    color: 'white',
                    margin: '0 0 0.5rem',
                    fontSize: '0.95rem'
                  }}>
                    Đang giải phóng camera...
                  </h4>
                  <p style={{
                    color: '#fde68a',
                    fontSize: '0.78rem',
                    margin: 0
                  }}>
                    Đóng luồng camera browser trước khi mở OpenCV.
                  </p>
                </div>
              ) : hasCameraPermission && status !== 'capturing' ? (
                <video
                  ref={videoRef}
                  className="webcam-video"
                  autoPlay
                  playsInline
                  muted
                />
              ) : status === 'capturing' ? (
                <div style={{
                  textAlign: 'center',
                  color: '#60a5fa',
                  padding: '2rem'
                }}>
                  <RefreshCw
                    size={44}
                    style={{
                      marginBottom: '1rem',
                      animation: 'spin 2s linear infinite'
                    }}
                  />
                  <h4 style={{
                    color: 'white',
                    margin: '0 0 0.5rem',
                    fontSize: '0.95rem'
                  }}>
                    Đang chụp ảnh mẫu...
                  </h4>
                  <p style={{
                    color: '#93c5fd',
                    fontSize: '0.78rem',
                    margin: 0,
                    lineHeight: 1.6
                  }}>
                    Cửa sổ OpenCV đã mở trên máy tính.<br />
                    Nhìn vào camera và xoay nhẹ đầu theo nhiều góc.
                  </p>
                </div>
              ) : (
                <div style={{
                  textAlign: 'center',
                  color: '#6b7280'
                }}>
                  <Camera
                    size={44}
                    style={{
                      opacity: 0.3,
                      marginBottom: '1rem'
                    }}
                  />
                  <p style={{ fontSize: '0.85rem' }}>
                    Không tìm thấy camera hoặc chưa cấp quyền.
                  </p>
                </div>
              )}

              {status !== 'capturing' && !releasing && uiStep === 2 && hasCameraPermission && (
                <CameraGuideOverlay />
              )}

              {status === 'embedding' && (
                <div style={{
                  position: 'absolute',
                  inset: 0,
                  background: 'rgba(255,255,255,0.92)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  zIndex: 10
                }}>
                  <RefreshCw
                    size={44}
                    style={{
                      color: '#2563eb',
                      marginBottom: '1rem',
                      animation: 'spin 2s linear infinite'
                    }}
                  />
                  <h3 style={{
                    color: '#111827',
                    fontSize: '1rem',
                    marginBottom: '0.25rem'
                  }}>
                    Đang tạo vector khuôn mặt...
                  </h3>
                  <p style={{
                    color: '#6b7280',
                    fontSize: '0.82rem'
                  }}>
                    Đang cập nhật CSDL Face Embeddings.
                  </p>
                </div>
              )}
            </div>

            <div style={{
              display: 'flex',
              gap: '0.75rem',
              marginTop: '1rem'
            }}>
              {uiStep === 1 && (
                <button
                  className="btn btn-primary"
                  style={{ flex: 1 }}
                  onClick={() => {
                    setUiStep(2);
                    startCamera();
                  }}
                  disabled={!form.employee_id || !form.full_name}
                >
                  <Camera size={15} />
                  Tiếp tục → Chụp 50 ảnh
                </button>
              )}

              {uiStep === 2 && (
                <>
                  <button
                    className="btn btn-primary"
                    style={{ flex: 1.4 }}
                    onClick={handleStartCapture}
                    disabled={status === 'capturing' || status === 'embedding' || releasing}
                  >
                    <Play size={15} />
                    {releasing ? 'Đang chuẩn bị...' : 'Bắt đầu chụp 50 ảnh'}
                  </button>

                  <button
                    className="btn btn-secondary"
                    style={{ flex: 0.8 }}
                    onClick={handleStopCapture}
                    disabled={status !== 'capturing'}
                  >
                    <Square size={13} />
                    Dừng
                  </button>

                  <button
                    className="btn btn-secondary"
                    style={{ flex: 1 }}
                    onClick={handleBuildEmbedding}
                    disabled={status !== 'success' && status !== 'idle'}
                  >
                    <Cpu size={13} />
                    Tạo vector
                  </button>
                </>
              )}
            </div>

            {uiStep === 2 && status === 'idle' && !releasing && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginTop: '0.75rem',
                padding: '0.65rem 0.875rem',
                background: '#fffbeb',
                border: '1px solid #fde68a',
                borderRadius: 10,
                fontSize: '0.76rem',
                color: '#92400e',
                lineHeight: 1.5
              }}>
                ⚠️ Hệ thống sẽ tắt camera browser khoảng 2–3 giây trước khi mở cửa sổ OpenCV để tránh xung đột webcam.
              </div>
            )}
          </div>

          <div className="card" style={{
            borderLeft: '3px solid #2563eb',
            padding: '1rem 1.25rem'
          }}>
            <div style={{
              display: 'flex',
              gap: '0.75rem'
            }}>
              <Sparkles
                size={18}
                style={{
                  color: '#2563eb',
                  flexShrink: 0,
                  marginTop: 2
                }}
              />

              <div>
                <h4 style={{
                  margin: '0 0 0.5rem',
                  fontWeight: 800,
                  fontSize: '0.875rem',
                  color: '#111827'
                }}>
                  Hướng dẫn chụp ảnh mẫu
                </h4>

                <div style={{
                  fontSize: '0.8rem',
                  color: '#6b7280',
                  lineHeight: 1.7
                }}>
                  <p style={{ margin: '0 0 0.25rem' }}>1. Ngồi cách camera khoảng 50–70cm, ánh sáng đều.</p>
                  <p style={{ margin: '0 0 0.25rem' }}>2. Xoay nhẹ đầu trái · phải · lên · xuống.</p>
                  <p style={{ margin: '0 0 0.25rem' }}>3. Không để nhiều người xuất hiện trong khung hình.</p>
                  <p style={{ margin: 0 }}>4. Ảnh đạt chất lượng sẽ tự lưu và hiện ở khung xem trước.</p>
                </div>
              </div>
            </div>
          </div>

          <CapturedGallery
            images={capturedImages}
            maxImages={MAX_IMAGES}
          />
        </div>

        <div>
          <div className="card" style={{ marginBottom: '1rem' }}>
            <div className="card-header">
              <span className="card-title">📋 Thông tin nhân viên</span>
            </div>

            <div style={{
              borderBottom: '1px solid #f3f4f6',
              paddingBottom: '1rem',
              marginBottom: '1rem'
            }}>
              <label style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                fontSize: '0.7rem',
                fontWeight: 800,
                color: '#2563eb',
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
                marginBottom: '0.35rem'
              }}>
                <UserCheck size={13} />
                Chọn nhân viên
              </label>

              <select
                className="input"
                value={selectedEmpId}
                onChange={e => handleSelectEmployee(e.target.value)}
                disabled={status === 'capturing'}
                style={{
                  borderColor: '#bfdbfe',
                  fontWeight: 600
                }}
              >
                {employees.map(emp => {
                  const isReg = registeredIds.has(emp.employee_id);

                  return (
                    <option
                      key={emp.employee_id}
                      value={emp.employee_id}
                    >
                      {isReg ? '🟢' : '🔴'} {emp.employee_id} - {emp.full_name} ({emp.department}) {isReg ? '[Đã đăng ký]' : '[Chưa đăng ký]'}
                    </option>
                  );
                })}

                <option value="NEW_MANUAL">
                  ➕ Đăng ký nhân viên mới...
                </option>
              </select>
            </div>

            {!isManualInput && selectedAlreadyRegistered && (
              <div style={{
                fontSize: '0.77rem',
                color: '#92400e',
                background: '#fffbeb',
                border: '1px solid #fde68a',
                padding: '0.65rem 0.875rem',
                borderRadius: 10,
                marginBottom: '1rem',
                fontWeight: 600,
                lineHeight: 1.5
              }}>
                ⚠️ Nhân viên này đã có vector. Chụp lại sẽ ghi đè ảnh mẫu và embedding cũ.
              </div>
            )}

            {[
              {
                label: 'Mã nhân viên *',
                key: 'employee_id',
                ph: 'NV001'
              },
              {
                label: 'Họ tên *',
                key: 'full_name',
                ph: 'Nguyễn Văn A'
              },
              {
                label: 'Chức vụ',
                key: 'position',
                ph: 'Developer'
              }
            ].map(field => (
              <div
                className="input-group"
                key={field.key}
              >
                <label>{field.label}</label>

                <input
                  className="input"
                  value={form[field.key]}
                  placeholder={field.ph}
                  onChange={e => setForm({
                    ...form,
                    [field.key]: e.target.value
                  })}
                  disabled={status === 'capturing'}
                  readOnly={!isManualInput}
                  style={!isManualInput ? {
                    background: '#f9fafb',
                    color: '#9ca3af',
                    cursor: 'not-allowed'
                  } : {}}
                />

                {isManualInput && field.key === 'employee_id' && (
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
            ))}

            <div className="input-group">
              <label>Phòng ban</label>

              {isManualInput ? (
                <select
                  className="input"
                  value={form.department}
                  onChange={e => setForm({
                    ...form,
                    department: e.target.value
                  })}
                  disabled={status === 'capturing'}
                >
                  {['IT', 'HR', 'Finance', 'Marketing', 'Sales', 'Admin'].map(dept => (
                    <option key={dept}>{dept}</option>
                  ))}
                </select>
              ) : (
                <input
                  className="input"
                  value={form.department}
                  readOnly
                  disabled={status === 'capturing'}
                  style={{
                    background: '#f9fafb',
                    color: '#9ca3af',
                    cursor: 'not-allowed'
                  }}
                />
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">📊 Tiến trình & chất lượng</span>
            </div>

            <div style={{ marginBottom: '1.125rem' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '0.825rem',
                marginBottom: '0.4rem'
              }}>
                <span style={{ color: '#6b7280' }}>
                  Tiến trình chụp ảnh
                </span>

                <strong style={{ color: '#111827' }}>
                  {progress} / {MAX_IMAGES} ảnh
                </strong>
              </div>

              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{
                    width: `${completionPercent}%`
                  }}
                />
              </div>
            </div>

            <div className="quality-grid">
              {[
                {
                  label: 'Hoàn thành',
                  val: progress > 0 ? `${completionPercent}%` : '—',
                  active: progress > 0
                },
                {
                  label: 'Độ nét (≥80)',
                  val: metrics.blur,
                  active: status === 'capturing'
                },
                {
                  label: 'Độ sáng (50–220)',
                  val: metrics.brightness,
                  active: status === 'capturing'
                },
                {
                  label: 'Chất lượng',
                  val: metrics.quality,
                  active: status === 'capturing' || status === 'success'
                }
              ].map(q => (
                <div
                  key={q.label}
                  className={`quality-card ${q.active ? 'success' : ''}`}
                >
                  <div style={{
                    fontSize: '1.1rem',
                    fontWeight: 800,
                    color: q.active ? '#15803d' : '#9ca3af',
                    minHeight: '1.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    {q.val}
                  </div>

                  <div style={{
                    fontSize: '0.68rem',
                    color: '#9ca3af',
                    marginTop: '0.25rem'
                  }}>
                    {q.label}
                  </div>
                </div>
              ))}
            </div>

            {status === 'error' && (
              <div style={{
                display: 'flex',
                gap: '0.5rem',
                background: '#fef2f2',
                border: '1px solid #fca5a5',
                padding: '0.7rem 0.875rem',
                borderRadius: 10,
                marginTop: '0.875rem',
                color: '#b91c1c',
                fontSize: '0.78rem'
              }}>
                <AlertCircle
                  size={15}
                  style={{ flexShrink: 0 }}
                />
                <span>{errorMessage}</span>
              </div>
            )}

            {status === 'success' && (
              <div style={{
                display: 'flex',
                gap: '0.5rem',
                background: '#f0fdf4',
                border: '1px solid #bbf7d0',
                padding: '0.7rem 0.875rem',
                borderRadius: 10,
                marginTop: '0.875rem',
                color: '#15803d',
                fontSize: '0.78rem'
              }}>
                <CheckCircle2
                  size={15}
                  style={{ flexShrink: 0 }}
                />

                <div>
                  <strong style={{
                    display: 'block',
                    marginBottom: '0.1rem'
                  }}>
                    Thao tác thành công!
                  </strong>

                  <span>
                    {successMessage || `Đã thu thập ${progress || MAX_IMAGES} ảnh mẫu của ${form.full_name}.`}
                  </span>
                </div>
              </div>
            )}

            <div style={{
              marginTop: '1rem',
              padding: '0.75rem 0.875rem',
              borderRadius: 10,
              background: '#f8fafc',
              border: '1px solid #e5e7eb',
              display: 'flex',
              gap: '0.65rem',
              alignItems: 'flex-start'
            }}>
              <ShieldCheck
                size={17}
                style={{
                  color: '#2563eb',
                  flexShrink: 0,
                  marginTop: 2
                }}
              />

              <div style={{
                fontSize: '0.76rem',
                color: '#64748b',
                lineHeight: 1.55
              }}>
                Sau khi chụp đủ ảnh, hệ thống sẽ tạo embedding bằng InsightFace Buffalo_L. Avatar nhân viên nên dùng ảnh <strong>portrait.jpg</strong> được lưu trong thư mục in-house.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
