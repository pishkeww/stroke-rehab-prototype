# StrokeRehab: Computer Vision Prototype for Motor Recovery Monitoring

**StrokeRehab** is a webcam-based prototype designed to monitor and quantify movement symmetry and motor usage during post-stroke rehabilitation. 

The system leverages **MediaPipe Pose** to provide real-time feedback on range of motion and bilateral symmetry, transforming a standard laptop camera into a clinical monitoring tool.

---

## Core Features & Modules

The system breaks down rehabilitation monitoring into three key quantitative metrics:

* **Symmetry Analysis (`symmetry.py`)**: Compares bilateral joint angles and movement amplitudes to detect "learned non-use" or compensatory patterns in affected limbs.
* **Smoothness Metrics (`smoothness.py`)**: Quantifies the quality of motor control by analyzing the velocity profile and jitter of movement trajectories.
* **Real-time Pose Estimation (`poseestimation.py`)**: A high-performance wrapper for MediaPipe that extracts 33 skeletal landmarks at low latency.

---

##  Technical Highlights

* **Framework**: Python & MediaPipe (BlazePose GHUM model).
* **Accessibility**: Designed for standard RGB webcams, eliminating the need for expensive depth sensors or wearable markers.
* **Quantification**: Transitions from qualitative therapist observations to objective, numerical data points (angles, velocity, symmetry ratios).

---

## Future Development 
* **Clinical Validation**: Planned testing with post-stroke individuals to correlate CV-derived metrics with the **Fugl-Meyer Assessment (FMA)** scale.
* **Cross-Platform Deployment**: Expanding the prototype into a standalone Windows application for at-home tele-rehabilitation.
* **Advanced Denoising**: Integrating temporal smoothing (e.g., Savitzky-Golay filters) to improve landmark stability in varied lighting conditions.

---
*Note: This project is intended as a support tool for rehabilitation and is not a diagnostic or clinical assessment system. Validation on clinical populations is ongoing.*
