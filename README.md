#  Industrial Pick-and-Place Vision System

**Real-time object detection and robotic grasp decision-making using YOLOv8 and OpenCV.**

> Built as a portfolio project by **Lakkireddy Chakradhar Reddy**  
> MS Robotics & Autonomous Systems — Arizona State University  

---

## 🎯 Project Overview

This system simulates the vision pipeline of an **industrial pick-and-place robotic arm**. A webcam feed is processed in real time using **YOLOv8** to:

1. Detect objects in the scene
2. Compute the **centroid (gripper target)** of each detection
3. Map pixel coordinates → **simulated robot workspace (mm)**
4. Make a **grasp decision** based on object class, size, and workspace boundary
5. Display a live HUD with all relevant information

This directly mirrors the workflow used in the robotic arm automation project at **SP Hi-Tech Printers**, where vision-guided control logic was used for pick-and-place operations.

---

## 🖥️ Demo

```
┌────────────────────────────────────────────────┐
│ PICK & PLACE VISION    FPS: 28.4               │
│ Objects: 3                                     │
│ Graspable: 2  │  Non-graspable: 1              │
│ ─────────────────────────────────────────────  │
│ [1] bottle  Conf: 91%                          │
│     Pixel: (312, 240)                          │
│     Robot: (270.5 mm, 192.0 mm)               │
│     → GRASP ✓                                  │
│ [2] person  Conf: 87%                          │
│     → NOT GRASPABLE — class                   │
└────────────────────────────────────────────────┘
```

---

## ⚙️ Features

- ✅ **YOLOv8** (nano/small/medium) — fast inference, runs on CPU
- ✅ **Real-time grasp decision logic** (class, size, workspace boundary)
- ✅ **Pixel → Robot coordinate mapping** (ready for real calibration)
- ✅ **Live HUD panel** with FPS, per-object decisions, and coordinates
- ✅ **Workspace boundary visualization**
- ✅ **Keyboard controls**: quit (`q`), snapshot (`s`), pause (`p`)
- ✅ Works with webcam or any video file

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install ultralytics opencv-python numpy
```

### 2. Run with webcam
```bash
python vision_system.py
```

### 3. Other options
```bash
# Use a different webcam
python vision_system.py --source 1

# Use a video file
python vision_system.py --source demo.mp4

# Higher confidence threshold
python vision_system.py --conf 0.6

# Use larger (more accurate) model
python vision_system.py --model s
```

> **Note:** YOLOv8 weights (`yolov8n.pt`) download automatically on first run (~6 MB).

---

## 📁 Project Structure

```
pick_place_vision/
├── vision_system.py     # Main detection + decision pipeline
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🔧 How It Works

### Detection
YOLOv8 runs inference on each frame, returning bounding boxes, class labels, and confidence scores.

### Grasp Decision Logic
Each detected object goes through three checks:

| Check | Condition | Result |
|-------|-----------|--------|
| Class filter | Is it in `GRASPABLE_CLASSES`? | NOT GRASPABLE — class |
| Size filter | `MIN_GRASP_AREA < area < MAX_GRASP_AREA` | NOT GRASPABLE — too small/large |
| Workspace | Centroid inside workspace rectangle | OUT OF WORKSPACE |
| ✅ All pass | — | **GRASP ✓** |

### Coordinate Mapping
```python
robot_x = interp(pixel_cx, [0, frame_width],  [0, 500])  # mm
robot_y = interp(pixel_cy, [0, frame_height], [0, 400])  # mm
```
Replace with your actual homography/calibration matrix for real robot deployment.

---

## 🔌 Real Robot Integration

To connect this to a real arm (e.g., via ROS2 or serial):

```python
# In grasp_decision(), when decision == "GRASP ✓":
# Publish to ROS2 topic
ros_publisher.publish(PoseStamped(x=rx, y=ry, z=0))

# Or send over serial to Arduino/STM32
serial_port.write(f"MOVE {rx} {ry}\n".encode())
```

---

## 📦 Requirements

```
ultralytics>=8.0
opencv-python>=4.8
numpy>=1.24
```

---

## 👤 Author

**Lakkireddy Chakradhar Reddy**  
[LinkedIn](https://linkedin.com) | [GitHub](https://github.com)  
chakrireddy681@gmail.com

MS Robotics & Autonomous Systems, Arizona State University  
GPA: 4.0 | Expected Graduation: December 2026
