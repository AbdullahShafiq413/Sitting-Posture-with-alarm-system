# Sitting Posture Detector with Alarm System

A real-time **sitting posture detection** app built with **YOLOv5 + PyQt5**.  
It monitors your webcam and, if you **slouch for more than 30 seconds**, it triggers:

- 🔊 A beep sound  
- 🗣️ A voice reminder  
- 🟥 A red on-screen warning message  

to help you keep a healthy sitting posture.

---

## 🚀 Features

- YOLOv5 model for `sitting_good` / `sitting_bad`
- Live webcam preview with brightness & contrast controls
- FPS, confidence, and class information overlay
- 30-second slouch timer with audio + visual alerts
- Flip, rotate, and fullscreen options for the camera feed

---

## ⚙️ Installation (Windows)

1. **Clone the repository**

   ```bash
   Open Project in VS Code (or any other)
   Go to terminal 
   cd Sitting-Posture-with-alarm-system
````

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**

   * For **CPU only**:

     ```bash
     pip install -r requirements_windows.txt
     ```

   * For **NVIDIA GPU + CUDA**:

     ```bash
     pip install -r requirements_windows_gpu.txt
     ```

---

## ▶️ Usage

```bash
python application.py
```

* Select your camera (e.g. *HP HD Camera*).
* Click **Start** to begin detection.
* If you stay in `sitting_bad` posture for > **30 seconds**, the alarm system will warn you.

---

## 📁 Project Structure

```text
application.py          # Entry point (PyQt app)
app_models/             # Model & YOLO inference
app_views/              # GUI and worker threads
app_controllers/        # App logic and utilities
data/                   # Models, icons, and assets
requirements_windows*.txt
```

