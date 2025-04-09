<div align="center">
  <h1>Face Capture, Liveness Check, Attributes Online Demo (Next.js)</h1>
  <img src="https://miniai.live/wp-content/uploads/2024/02/logo_name-1-768x426-1.png" alt="MiniAiLive Logo" width="300">
  <p>Enterprise-grade on-premise biometric solutions demo</p>
</div>

## Overview

This repository contains a Next.js demo application showcasing MiniAiLive's complete suite of on-premise biometric SDKs:

- **Face Detection SDK**
- **Face Liveness Detection SDK**
- **Face Attributes SDK**

![Demo Screenshot](https://github.com/user-attachments/assets/ed6b5e04-f5bc-4bff-9d1e-4e9d88720869)

## Key Features

✔ **100% On-Premise Processing** - All biometric processing occurs on your servers  
✔ **Zero Data Exfiltration** - No sensitive data leaves your infrastructure  
✔ **Enterprise-Ready** - Production-grade SDKs with commercial licensing  
✔ **Modern Web Stack** - Built with Next.js for optimal performance  

## Getting Started

### Prerequisites
- Node.js 18+
- npm/yarn/pnpm/bun

### Installation
#### Frontend Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MiniAiLive/FaceCapture-LivenessDetection-Web-Next.JS.git
2. Install dependencies:
   ```bash
   cd frontend
   npm install
3. Run the development server:
   ```bash
   npm run dev
4. Open http://localhost:3000 in your browser
#### Backend Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MiniAiLive/FaceCapture-LivenessDetection-Web-Next.JS.git
2. Install dependencies: (Recommend: Python Environment Version = 3.10.x)
   ```bash
   cd backend
   pip install -r requirements.txt
3. Run the development server:
   ```bash
   python app.py
4. Backend API will run on http://localhost:3001. Check API with this endpoint.
   POST Request: http://127.0.0.1:3001/api/detect
   Body: JSON Raw data, Base64 image data

## Deployment - Vercel (Recommended)
[Deploy with Vercel](https://vercel.com/new/clone?repository-url=https%253A%252F%252Fgithub.com%252FMiniAiLive%252Ffacesdk-idsdk-online-demo-nextjs)

## Documentation
For full SDK documentation and integration guides, visit [here](https://docs.miniai.live)

## Request license
If you want to test it on your own server, feel free to [Contact US](https://www.miniai.live/contact/) to get a trial License. We are 24/7 online on [WhatsApp](https://wa.me/+19162702374).


## Face & IDSDK Online Demo, Resources
<div style="display: flex; justify-content: center; align-items: center;"> 
   <table style="text-align: center;">
      <tr>
         <td style="text-align: center; vertical-align: middle;"><a href="https://github.com/MiniAiLive"><img src="https://miniai.live/wp-content/uploads/2024/10/new_git-1-300x67.png" style="height: 60px; margin-right: 5px;" title="GITHUB"/></a></td> 
         <td style="text-align: center; vertical-align: middle;"><a href="https://huggingface.co/MiniAiLive"><img src="https://miniai.live/wp-content/uploads/2024/10/new_hugging-1-300x67.png" style="height: 60px; margin-right: 5px;" title="HuggingFace"/></a></td> 
         <td style="text-align: center; vertical-align: middle;"><a href="https://demo.miniai.live"><img src="https://miniai.live/wp-content/uploads/2024/10/new_gradio-300x67.png" style="height: 60px; margin-right: 5px;" title="Gradio"/></a></td> 
      </tr> 
      <tr>
         <td style="text-align: center; vertical-align: middle;"><a href="https://docs.miniai.live/"><img src="https://miniai.live/wp-content/uploads/2024/10/a-300x70.png" style="height: 60px; margin-right: 5px;" title="Documentation"/></a></td> 
         <td style="text-align: center; vertical-align: middle;"><a href="https://www.youtube.com/@miniailive"><img src="https://miniai.live/wp-content/uploads/2024/10/Untitled-1-300x70.png" style="height: 60px; margin-right: 5px;" title="Youtube"/></a></td> 
         <td style="text-align: center; vertical-align: middle;"><a href="https://play.google.com/store/apps/dev?id=5831076207730531667"><img src="https://miniai.live/wp-content/uploads/2024/10/googleplay-300x62.png" style="height: 60px; margin-right: 5px;" title="Google Play"/></a></td>
      </tr>
   </table>
</div>

## Our Products

### Face Recognition SDK
| No | Project | Features |
|----|---------|-----------|
| 1 | [FaceRecognition-SDK-Docker](https://github.com/MiniAiLive/FaceRecognition-SDK-Docker) | 1:1 & 1:N Face Matching SDK |
| 2 | [FaceRecognition-SDK-Windows](https://github.com/MiniAiLive/FaceRecognition-SDK-Windows) | 1:1 & 1:N Face Matching SDK |
| 3 | [FaceRecognition-SDK-Linux](https://github.com/MiniAiLive/FaceRecognition-SDK-Linux) | 1:1 & 1:N Face Matching SDK |
| 4 | [FaceRecognition-LivenessDetection-SDK-Android](https://github.com/MiniAiLive/FaceRecognition-LivenessDetection-SDK-Android) | 1:1 & 1:N Face Matching, 2D & 3D Face Passive Liveness Detection SDK |
| 5 | [FaceRecognition-LivenessDetection-SDK-iOS](https://github.com/MiniAiLive/FaceRecognition-LivenessDetection-SDK-iOS) | 1:1 & 1:N Face Matching, 2D & 3D Face Passive Liveness Detection SDK |
| 6 | [FaceRecognition-LivenessDetection-SDK-CPP](https://github.com/MiniAiLive/FaceRecognition-LivenessDetection-SDK-CPP) | 1:1 & 1:N Face Matching, 2D & 3D Face Passive Liveness Detection SDK |
| 7 | [FaceMatching-SDK-Android](https://github.com/MiniAiLive/FaceMatching-SDK-Android) | 1:1 Face Matching SDK |
| 8 | [FaceAttributes-SDK-Android](https://github.com/MiniAiLive/FaceAttributes-SDK-Android) | Face Attributes, Age & Gender Estimation SDK |

### Face Liveness Detection SDK
| No | Project | Features |
|----|---------|-----------|
| 1 | [FaceLivenessDetection-SDK-Docker](https://github.com/MiniAiLive/FaceLivenessDetection-SDK-Docker) | 2D & 3D Face Passive Liveness Detection SDK |
| 2 | [FaceLivenessDetection-SDK-Windows](https://github.com/MiniAiLive/FaceLivenessDetection-SDK-Windows) | 2D & 3D Face Passive Liveness Detection SDK |
| 3 | [FaceLivenessDetection-SDK-Linux](https://github.com/MiniAiLive/FaceLivenessDetection-SDK-Linux) | 2D & 3D Face Passive Liveness Detection SDK |
| 4 | [FaceLivenessDetection-SDK-Android](https://github.com/MiniAiLive/FaceLivenessDetection-SDK-Android) | 2D & 3D Face Passive Liveness Detection SDK |
| 5 | [FaceLivenessDetection-SDK-iOS](https://github.com/MiniAiLive/FaceLivenessDetection-SDK-iOS) | 2D & 3D Face Passive Liveness Detection SDK |

### ID Document Recognition SDK
| No | Project | Features |
|----|---------|-----------|
| 1 | [ID-DocumentRecognition-SDK-Docker](https://github.com/MiniAiLive/ID-DocumentRecognition-SDK-Docker) | ID Document, Passport, Driver License, Credit Card, MRZ Recognition SDK |
| 2 | [ID-DocumentRecognition-SDK-Windows](https://github.com/MiniAiLive/ID-DocumentRecognition-SDK-Windows) | ID Document, Passport, Driver License, Credit Card, MRZ Recognition SDK |
| 3 | [ID-DocumentRecognition-SDK-Linux](https://github.com/MiniAiLive/ID-DocumentRecognition-SDK-Linux) | ID Document, Passport, Driver License, Credit Card, MRZ Recognition SDK |
| 4 | [ID-DocumentRecognition-SDK-Android](https://github.com/MiniAiLive/ID-DocumentRecognition-SDK-Android) | ID Document, Passport, Driver License, Credit Card, MRZ Recognition SDK |

### ID Document Liveness Detection SDK
| No | Project | Features |
|----|---------|-----------|
| 1 | [ID-DocumentLivenessDetection-SDK-Docker](https://github.com/MiniAiLive/ID-DocumentLivenessDetection-SDK-Docker) | ID Document Liveness Detection SDK |
| 2 | [ID-DocumentLivenessDetection-SDK-Windows](https://github.com/MiniAiLive/ID-DocumentLivenessDetection-SDK-Windows) | ID Document Liveness Detection SDK |
| 3 | [ID-DocumentLivenessDetection-SDK-Linux](https://github.com/MiniAiLive/ID-DocumentLivenessDetection-SDK-Linux) | ID Document Liveness Detection SDK |

### Web & Desktop Demo
| No | Project | Features |
|----|---------|-----------|
| 1 | [FaceRecognition-IDRecognition-Playground-Next.JS](https://github.com/MiniAiLive/FaceRecognition-IDRecognition-Playground-Next.JS) | FaceSDK & IDSDK Playground |
| 2 | [FaceCapture-LivenessDetection-Next.JS](https://github.com/MiniAiLive/FaceCapture-LivenessDetection-Next.JS) | Face Capture, Face LivenessDetection, Face Attributes |
| 3 | [FaceMatching-Windows-App](https://github.com/MiniAiLive/FaceMatching-Windows-App) | 1:1 Face Matching Windows Demo Application |


## About MiniAiLive
[MiniAiLive](https://www.miniai.live/) is a leading AI solutions company specializing in computer vision and machine learning technologies. We provide cutting-edge solutions for various industries, leveraging the power of AI to drive innovation and efficiency.

## Contact US
For any inquiries or questions, please contact us on [WhatsApp](https://wa.me/+19162702374).
