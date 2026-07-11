# ExerciseIQ 🏋️
> Real-time squat analyser using computer vision and pose estimation.

ExerciseIQ uses your webcam and MediaPipe's pose landmark model to analyse squat form in real time — detecting knee alignment, counting reps, and giving motivational feedback as you train.

---

## Demo

> Stand sideways to your camera (right side facing lens) and start squatting.

![ExerciseIQ Demo](demo.gif)
<!-- Record a screen capture and add it here -->![Uploading exercise-ezgif.com-video-to-gif-converter.gif…]()


---

## How It Works

1. **Pose Detection** — MediaPipe's `PoseLandmarker` model detects 33 body landmarks per frame
2. **Angle Calculation** — Computes knee angle using dot product between hip→knee and ankle→knee vectors
3. **Knee-Over-Toe Detection** — Draws a vertical reference line at the toe tip. If knee crosses this line for more than 0.3 seconds, flags bad form
4. **Smoothing** — Rolling average over 5 frames eliminates jitter from pixel-level noise
5. **Rep Counter** — Counts reps by tracking stage transitions: DOWN (angle < 90°) → UP (angle > 160°)
6. **Form-Gated Counting** — Only counts a rep as "good" if knee stayed behind toe line throughout

---

## Features

- 🦵 Real-time knee angle display
- 📏 Vertical toe reference line — visual boundary for knee travel
- ✅ Good rep / ❌ Bad rep counter separated
- ⏱️ 0.3s jitter filter — ignores momentary pixel fluctuation
- 💬 Motivational feedback on every rep
- 🟢 Green skeleton overlay with highlighted knee tracking dot

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| OpenCV | Webcam capture and frame rendering |
| MediaPipe | Pose landmark detection |
| NumPy | Vector math for angle calculation |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Asterisk4589/ExerciseIQ.git
cd ExerciseIQ
```

**2. Install dependencies**
```bash
pip install opencv-python mediapipe numpy
```

**3. Run**
```bash
python main.py
```

The pose model (`pose_landmarker_full.task`) downloads automatically on first run (~30MB).

---

## Usage

- Stand **sideways** to your camera — right knee facing the lens
- Make sure your **full body is visible** from head to toe
- Do squats — the system tracks form in real time
- Press **Q** to quit and see your session summary in terminal

---

## Landmark Indices Used

| Body Part | MediaPipe Index |
|-----------|----------------|
| Right Hip | 24 |
| Right Knee | 26 |
| Right Ankle | 28 |
| Right Toe | 32 |

---

## Roadmap

- [ ] Session summary screen on quit (good %, main issue detected)
- [ ] Support for left-side camera view
- [ ] Additional exercises — lunges, deadlifts
- [ ] Angle timeline graph — visualise form breakdown per rep
- [ ] DanceIQ mode — extend to dance movement analysis

---

## Why I Built This

Most people squat with bad form and don't know it. A trainer costs money. A mirror doesn't give feedback. This tool gives real-time, specific correction — "your knee is crossing your toe" — so anyone can self-correct without a coach.

---

## Author

**MJ** — CS Student, 3rd Year | [GitHub](https://github.com/Asterisk4589)

---

*ExerciseIQ v0.1 — built in one day using MediaPipe and OpenCV*
