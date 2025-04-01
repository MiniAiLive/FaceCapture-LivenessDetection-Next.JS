import { useState, useRef, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

export default function FaceCaptureDemo() {
  const webcamRef = useRef(null);
  const [hasPermission, setHasPermission] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [imgSrc, setImgSrc] = useState(null);
  const [isCaptured, setIsCaptured] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [faceData, setFaceData] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [expandedFace, setExpandedFace] = useState(null);
  const [notification, setNotification] = useState(null);

    // Dark theme colors
  const theme = {
    background: '#121212',
    surface: '#1e1e1e',
    primary: '#BB86FC',
    secondary: '#03DAC6',
    text: '#e1e1e1',
    textSecondary: '#a0a0a0',
    error: '#CF6679',
    success: '#4CAF50',
    warning: '#FFC107'
  };
    
  // Check camera permissions
  useEffect(() => {
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        setHasPermission(true);
        stream.getTracks().forEach(track => track.stop());
      } catch (err) {
        console.error("Camera error:", err);
        setHasPermission(false);
        setCameraError(err.message);
      }
    })();
  }, []);

  // Auto-hide notifications after 5 seconds
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => {
        setNotification(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: "user",
    frameRate: { ideal: 60, max: 60 }
  };

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImgSrc(imageSrc);
    setIsCaptured(true);
    setFaceData(null);
    setApiError(null);
    setExpandedFace(null);
  }, [webcamRef]);

  const retake = () => {
    setImgSrc(null);
    setIsCaptured(false);
    setFaceData(null);
    setApiError(null);
    setExpandedFace(null);
  };

  const getAttributes = async () => {
    if (!imgSrc) return;
    
    setIsProcessing(true);
    setApiError(null);
    setNotification(null);
    
    try {
      const base64Image = imgSrc.split(',')[1];
      
      const response = await axios.post('http://localhost:3001/api/detect', {
        image: base64Image
      }, {
        headers: {
          'Content-Type': 'application/json'
            }
        });

      if (response.data.faces && response.data.faces.length > 0) {
        setFaceData(response.data);
        setNotification({
          type: 'success',
          message: `Detected ${response.data.faces.length} face(s)`
        });
      } else {
        setNotification({
          type: 'warning',
          message: 'No faces detected in the image'
        });
      }
    } catch (error) {
      console.error("API Error:", error);
      const errorMessage = error.response?.data?.error || 
                         error.message || 
                         'Failed to process face attributes';
      setNotification({
        type: 'error',
        message: errorMessage
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleFaceDetails = (faceIndex) => {
    setExpandedFace(expandedFace === faceIndex ? null : faceIndex);
  };

  if (hasPermission === false) {
    return (
      <div style={{ 
        padding: '2rem', 
        textAlign: 'center',
        maxWidth: '600px',
        margin: '0 auto',
        backgroundColor: theme.background,
        minHeight: '100vh',
        color: theme.text
      }}>
        <h2>Camera Access Required</h2>
        <p style={{ color: theme.error }}>{cameraError || 'Camera permission was denied.'}</p>
        <ol style={{ textAlign: 'left', marginTop: '1rem', color: theme.text }}>
          <li>Check your browser's permission settings</li>
          <li>Ensure no other app is using the camera</li>
          <li>Try refreshing the page</li>
          <li>If using HTTP, try HTTPS (required by some browsers)</li>
        </ol>
      </div>
    );
  }

  return (
    <div style={{
      width: '100%',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '2rem',
      backgroundColor: theme.background,
      position: 'relative',
      color: theme.text
    }}>
      {/* Notification System */}
      {notification && (
        <div style={{
          position: 'fixed',
          top: '20px',
          right: '20px',
          padding: '1rem',
          backgroundColor: notification.type === 'error' ? theme.error : 
                          notification.type === 'warning' ? theme.warning : theme.success,
          color: '#121212',
          borderRadius: '4px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.3)',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          maxWidth: '400px'
        }}>
          <span style={{ marginRight: '10px' }}>
            {notification.type === 'error' ? '❌' : 
             notification.type === 'warning' ? '⚠️' : '✅'}
          </span>
          <span>{notification.message}</span>
          <button 
            onClick={() => setNotification(null)}
            style={{
              marginLeft: '15px',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: '#121212',
              fontWeight: 'bold'
            }}
          >
            ×
          </button>
        </div>
      )}

      <h1 style={{ 
        color: theme.primary,
        marginBottom: '1.5rem',
        fontSize: '2rem',
        fontWeight: '500'
      }}>
        Face Detection System
      </h1>
      
      <div style={{
        display: 'flex',
        flexDirection: 'row',
        gap: '2rem',
        width: '100%',
        maxWidth: '1200px',
        justifyContent: 'center',
        alignItems: 'flex-start',
        flexWrap: 'wrap'
      }}>
        {/* Camera/Image Column */}
        <div style={{
          position: 'relative',
          width: '100%',
          maxWidth: '720px',
          margin: '2rem 0',
          backgroundColor: theme.surface,
          borderRadius: '8px',
          overflow: 'hidden'
        }}>
          {!isCaptured ? (
            <>
              <Webcam
                audio={false}
                ref={webcamRef}
                videoConstraints={videoConstraints}
                screenshotFormat="image/jpeg"
                style={{
                  display: 'block',
                  width: '100%',
                  height: 'auto'
                }}
                onUserMediaError={(error) => {
                  console.error("Webcam Error:", error);
                  setCameraError("Camera error: " + error.message);
                  setHasPermission(false);
                }}
              />
              
              {!isCaptured && (
                <div style={{
                  position: 'absolute',
                  bottom: '20px',
                  left: '0',
                  right: '0',
                  textAlign: 'center',
                  backgroundColor: 'rgba(30, 30, 30, 0.7)',
                  padding: '12px',
                  margin: '0 20px',
                  borderRadius: '4px'
                }}>
                  <p style={{ 
                    margin: 0, 
                    color: theme.text,
                    fontWeight: '500',
                    fontSize: '1rem'
                  }}>
                    Please capture image to detect face and get attributes.
                  </p>
                </div>
              )}
            </>
          ) : (
            <div style={{
              width: '100%',
              height: 'auto',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center'
            }}>
              <img 
                src={imgSrc} 
                alt="Captured face" 
                style={{
                  width: '100%',
                  height: 'auto',
                  display: 'block'
                }} 
              />
            </div>
          )}
        </div>

        {/* Results Column */}
        <div style={{
          width: '100%',
          maxWidth: '400px',
          backgroundColor: theme.surface,
          borderRadius: '8px',
          padding: '1.5rem',
          boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
          marginTop: '2rem',
          minHeight: '300px'
        }}>
          <h2 style={{ 
            color: theme.primary,
            marginTop: 0,
            marginBottom: '1.5rem',
            fontSize: '1.5rem'
          }}>
            Detection Results
          </h2>
          
          {isProcessing && (
            <div style={{
              padding: '1rem',
              backgroundColor: '#2a2a2a',
              borderRadius: '4px',
              marginBottom: '1rem',
              textAlign: 'center'
            }}>
              <p style={{ 
                color: theme.text,
                fontWeight: '500',
                margin: 0
              }}>
                Analyzing face attributes...
              </p>
            </div>
          )}

          {faceData && faceData.faces.length > 0 ? (
            <div style={{ marginTop: '1rem' }}>
              {faceData.faces.map((face) => (
                <div key={face.faceIndex} style={{
                  marginBottom: '1rem',
                  padding: '1rem',
                  backgroundColor: '#2a2a2a',
                  borderRadius: '8px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
                }}>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '1rem',
                    marginBottom: '0.5rem'
                  }}>
                    <img 
                      src={`data:image/jpeg;base64,${face.face}`} 
                      alt={`Face ${face.faceIndex}`}
                      style={{
                        width: '60px',
                        height: '60px',
                        borderRadius: '4px',
                        objectFit: 'cover'
                      }}
                    />
                    <div>
                      <p style={{ 
                        margin: 0, 
                        fontWeight: '500',
                        color: theme.text 
                      }}>
                        Person {face.faceIndex}
                      </p>
                      <p style={{ 
                        margin: 0,
                        color: theme.textSecondary,
                        fontSize: '0.9rem'
                      }}>
                        Age: {face.age}
                      </p>
                    </div>
                    <button
                      onClick={() => toggleFaceDetails(face.faceIndex)}
                      style={{
                        marginLeft: 'auto',
                        padding: '0.25rem 0.75rem',
                        backgroundColor: expandedFace === face.faceIndex ? theme.secondary : 'transparent',
                        color: expandedFace === face.faceIndex ? '#121212' : theme.secondary,
                        border: `1px solid ${theme.secondary}`,
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.8rem',
                        fontWeight: '500',
                        transition: 'all 0.2s ease'
                      }}
                    >
                      {expandedFace === face.faceIndex ? 'Hide' : 'Details'}
                    </button>
                  </div>

                  {expandedFace === face.faceIndex && (
                    <div style={{ 
                      marginTop: '1rem',
                      padding: '0.75rem',
                      backgroundColor: '#252525',
                      borderRadius: '4px'
                    }}>
                      <div style={{
                        display: 'grid',
                        gridTemplateColumns: '100px 1fr',
                        gap: '0.5rem',
                        alignItems: 'center'
                      }}>
                        <p style={{ margin: '0.25rem 0', color: theme.textSecondary }}>Gender:</p>
                        <p style={{ margin: '0.25rem 0', color: theme.text }}>{face.gender}</p>
                        
                        <p style={{ margin: '0.25rem 0', color: theme.textSecondary }}>Liveness:</p>
                        <p style={{ 
                          margin: '0.25rem 0',
                          color: face.liveness === 'Real' ? theme.success : theme.error
                        }}>
                          {face.liveness}
                        </p>
                        
                        <p style={{ margin: '0.25rem 0', color: theme.textSecondary }}>Emotion:</p>
                        <p style={{ margin: '0.25rem 0', color: theme.text }}>{face.emotion}</p>
                        
                        <p style={{ margin: '0.25rem 0', color: theme.textSecondary }}>Mask:</p>
                        <div>
                          <p style={{ 
                            margin: '0.25rem 0',
                            color: face.mask.status === 'Mask' ? theme.success : theme.error
                          }}>
                            {face.mask.status} ({Math.round(face.mask.confidence * 100)}%)
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p style={{ 
              color: theme.textSecondary, 
              textAlign: 'center',
              padding: '1rem'
            }}>
              {isProcessing ? 'Processing...' : 'No faces detected yet'}
            </p>
          )}
        </div>
      </div>

      <div style={{ 
        display: 'flex', 
        gap: '1rem', 
        marginTop: '1.5rem',
        width: '100%',
        maxWidth: '720px',
        justifyContent: 'center'
      }}>
        {!isCaptured ? (
          <button
            onClick={capture}
            style={{
              padding: '12px 24px',
              backgroundColor: theme.primary,
              color: '#121212',
              border: 'none',
              borderRadius: '4px',
              fontSize: '1rem',
              cursor: 'pointer',
              boxShadow: `0 2px 4px rgba(0,0,0,0.2)`,
              minWidth: '180px',
              fontWeight: '500',
              transition: 'all 0.2s ease',
              ':hover': {
                backgroundColor: '#a370d8',
                transform: 'translateY(-1px)'
              }
            }}
          >
            Capture Image
          </button>
        ) : (
          <>
            <button
              onClick={retake}
              style={{
                padding: '12px 24px',
                backgroundColor: 'transparent',
                color: theme.error,
                border: `1px solid ${theme.error}`,
                borderRadius: '4px',
                fontSize: '1rem',
                cursor: 'pointer',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                minWidth: '180px',
                fontWeight: '500',
                transition: 'all 0.2s ease'
              }}
            >
              Retake
            </button>
            <button
              onClick={getAttributes}
              disabled={isProcessing}
              style={{
                padding: '12px 24px',
                backgroundColor: isProcessing ? '#3a3a3a' : theme.secondary,
                color: isProcessing ? theme.textSecondary : '#121212',
                border: 'none',
                borderRadius: '4px',
                fontSize: '1rem',
                cursor: isProcessing ? 'not-allowed' : 'pointer',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                minWidth: '180px',
                fontWeight: '500',
                transition: 'all 0.2s ease'
              }}
            >
              {isProcessing ? 'Processing...' : 'Analyze Faces'}
            </button>
          </>
        )}
      </div>
    </div>
  );
}