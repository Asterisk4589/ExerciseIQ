# 🧠 ExerciseIQ
### Real-Time Human Movement Analysis Platform

> A modular Computer Vision and Biomechanics platform for intelligent exercise analysis, movement quality assessment, and real-time coaching feedback.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
---

## Demo

> Stand sideways to your camera (right side facing lens) and start squatting.

![ExerciseIQ Demo](demo.gif)
<!-- Record a screen capture and add it here -->!<img width="800" height="450" alt="demo" src="https://github.com/user-attachments/assets/bcbfa827-03e6-44f9-9a68-cf066b18149f" />



---

## Overview

ExerciseIQ is a modular Human Movement Analysis system that combines **Computer Vision**, **Biomechanics**, and **Exercise Intelligence** to evaluate exercise quality in real time.

Instead of acting as a simple repetition counter, ExerciseIQ extracts biomechanical features from human pose landmarks, analyzes movement quality, provides real-time coaching feedback, and generates post-session performance analytics.

The project is designed as a scalable software architecture where additional exercises can be integrated without modifying the core biomechanics engine.

---

## Motivation

Most exercise tracking applications only count repetitions.

ExerciseIQ aims to understand **how** an exercise is performed.

The long-term vision is to build an AI-powered movement assessment platform capable of:

- Exercise Quality Assessment
- Human Motion Analysis
- Personalized Coaching
- Performance Analytics
- Machine Learning-based Movement Scoring
- Injury Risk Indicators
- Long-term Progress Tracking

---

# Architecture

```

Camera / Video Input
│
▼
Pose Detection Layer
(MediaPipe)
│
▼
Biomechanics Engine
│
▼
Exercise Intelligence
│
▼
Analytics Engine
│
▼
Visualization & Reports

```

The project follows a layered architecture to maximize modularity, maintainability, and scalability.

---

# Current Features

### Pose Detection

- Real-time pose estimation
- Human skeletal landmark extraction
- Joint localization
- Landmark visualization

### Biomechanics Engine

- Joint angle computation
- Angular velocity calculation
- Range of Motion (ROM)
- Movement phase detection
- Exercise timing
- Temporal feature extraction

### Exercise Intelligence

- Squat repetition counting
- Exercise state detection
- Form validation
- Rule-based movement analysis
- Real-time coaching feedback

### Analytics

- Session statistics
- Rep quality tracking
- Performance visualization
- Knee angle timeline
- Session report generation

---

# Current Project Structure

```

ExerciseIQ/

├── main.py

├── config.py

├── biomechanics.py

├── pose/

│ ├── pose_detector.py

│ ├── landmarks.py

│ └── utils.py

├── intelligence/

│ ├── squat_analyzer.py

│ ├── rep_counter.py

│ ├── feedback_engine.py

│ └── score_engine.py

├── analytics/

│ ├── report_generator.py

│ ├── graph_generator.py

│ └── session.py

├── ui/

├── tests/

└── assets/

```

---

# Technologies Used

### Languages

- Python

### Computer Vision

- OpenCV
- MediaPipe

### Scientific Computing

- NumPy

### Visualization

- Matplotlib

### Development

- Git
- GitHub

---

# Current Workflow

```

Capture Video

↓

Pose Estimation

↓

Landmark Extraction

↓

Biomechanical Feature Computation

↓

Exercise Analysis

↓

Movement Quality Assessment

↓

Feedback Generation

↓

Session Analytics

↓

Performance Report

```

---

# Roadmap

## Phase 1 ✅

- Modular architecture
- Squat analysis
- Biomechanics engine
- Session analytics

## Phase 2 🚧

- Push-up analyzer
- Lunge analyzer
- Plank analyzer
- Deadlift analyzer

## Phase 3

- Personalized exercise profiles
- Adaptive feedback engine
- Multi-exercise sessions

## Phase 4

- Machine Learning movement assessment
- Automatic exercise recognition
- Personalized AI coaching

## Phase 5

- Web dashboard
- Mobile support
- Cloud deployment
- REST API
- Performance history
- User authentication

---

# Software Design Principles

ExerciseIQ follows modern software engineering practices:

- Modular Architecture
- Separation of Concerns
- Reusable Components
- Layered Design
- Extensible System Design
- Maintainable Codebase
- Scalable Exercise Framework

---

# Future Machine Learning Integration

The current version uses rule-based biomechanical analysis.

Future releases aim to incorporate supervised machine learning for:

- Exercise quality prediction
- Movement classification
- Personalized performance scoring
- Fatigue estimation
- Intelligent coaching recommendations

---

# Why ExerciseIQ?

Unlike traditional exercise trackers that only count repetitions, ExerciseIQ is designed as a **Human Movement Intelligence Platform** capable of understanding movement quality through Computer Vision and biomechanical analysis.

The long-term objective is to build a scalable framework that supports multiple exercises, interpretable movement analytics, and AI-assisted coaching.

---

# Author

**Maheep Singh**

Computer Science Undergraduate

Interested in:

- Software Engineering
- Computer Vision
- Artificial Intelligence
- Machine Learning
- Human Movement Analysis
- System Design

---

⭐ If you found this project interesting, consider giving it a star!
