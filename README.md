# DigiMark
Automated attendance system using computer vision and neural networks
# Automated Attendance System using Computer Vision

This project is a real-time attendance system that uses computer vision and neural networks to detect and recognize faces from video input and mark attendance automatically.

The system supports **live webcam feeds**, **phone camera streams**, and **uploaded video files**, and displays real-time results through a graphical user interface.

---

## Features

- Face-based attendance marking using computer vision
- Supports:
  - Live webcam input
  - Phone camera stream (IP camera)
  - Uploaded video files
- Real-time face detection, tracking, and recognition
- Attendance marked only after consistent presence
- Visual bounding boxes and names displayed on screen
- Attendance saved automatically as CSV files
- Multithreaded processing for smoother performance
- Optional CUDA-based optimizations (falls back to CPU if unavailable)

---

## How It Works

1. Known faces are stored as images inside the `known_faces/` folder.
2. Video frames are captured continuously from the selected source.
3. Faces are periodically detected and tracked across frames.
4. Face recognition runs in parallel using a thread pool.
5. A student is marked **Present** only after sustained detection.
6. Attendance is saved with timestamps in a CSV file.

---

## Project Structure
├── main.py
├── known_faces/
│ ├── person1.jpg
│ ├── person2.jpg
├── test_videos/
│ ├── sample.mp4
├── attendance_logs/
├── README.md
├── requirements.txt

---

## Technologies Used

- Python
- OpenCV
- face_recognition (dlib-based)
- NumPy
- Pandas
- Tkinter (GUI)
- Multithreading & ThreadPoolExecutor

---

## Performance & Optimization

- Detection runs at fixed time intervals instead of every frame
- Frames are downscaled before detection for speed
- Object trackers reduce repeated detections
- Face recognition runs asynchronously in parallel threads
- Attempts to use OpenCV CUDA acceleration if available (CPU fallback used otherwise)

> Note: GPU acceleration is optional and the system runs fully on CPU if CUDA is not available.

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Add reference images to the known_faces/ folder
    (Filename = Person Name)

3. Run the application

## Output

1. Attendance CSV files are saved in the attendance_logs/ folder
2. A summary window shows Present/Absent status at the end
