"""
Industrial Pick-and-Place Vision System
========================================
Author: Chakradhar Reddy
Description:
    Real-time object detection using YOLOv8 with webcam input.
    Computes gripper target coordinates and makes pick-and-place
    decisions based on object class and bounding box size.

Usage:
    python vision_system.py
    python vision_system.py --source 0          # webcam index
    python vision_system.py --source video.mp4  # video file
    python vision_system.py --conf 0.5          # confidence threshold

Requirements:
    pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import time
from ultralytics import YOLO


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Classes considered "graspable" by the industrial arm
GRASPABLE_CLASSES = {
    "bottle", "cup", "bowl", "banana", "apple", "orange",
    "sandwich", "book", "cell phone", "remote", "mouse",
    "keyboard", "scissors", "toothbrush", "vase", "box"
}

# Min/max bounding box area (px²) the arm can handle
MIN_GRASP_AREA = 2000
MAX_GRASP_AREA = 120000

# Simulated workspace limits (pixels → maps to real-world mm)
WORKSPACE = {"x_min": 100, "x_max": 540, "y_min": 50, "y_max": 430}

# Colors (BGR)
COLOR_GRASPABLE    = (0, 200, 50)    # green
COLOR_NOT_GRASPABLE = (0, 60, 220)   # red
COLOR_OUT_OF_WS    = (0, 165, 255)   # orange
COLOR_CROSSHAIR    = (255, 255, 0)   # cyan
COLOR_PANEL        = (20, 20, 20)    # dark panel


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def pixel_to_robot_coords(cx, cy, frame_w, frame_h,
                           robot_x_range=(0, 500),
                           robot_y_range=(0, 400)):
    """
    Map pixel coordinates to simulated robot workspace (mm).
    Replace with your actual calibration matrix for real deployment.
    """
    rx = np.interp(cx, [0, frame_w], robot_x_range)
    ry = np.interp(cy, [0, frame_h], robot_y_range)
    return round(rx, 1), round(ry, 1)


def grasp_decision(label, area, cx, cy):
    """
    Decide whether a human can grab the object.
    Returns (decision_str, color)
    """
    if label not in GRASPABLE_CLASSES:
        return "❌ Not graspable", COLOR_NOT_GRASPABLE
    if area < MIN_GRASP_AREA:
        return "⚠️ Too small to grab", COLOR_NOT_GRASPABLE
    if area > MAX_GRASP_AREA:
        return "⚠️ Too large to grab", COLOR_NOT_GRASPABLE
    if not (WORKSPACE["x_min"] < cx < WORKSPACE["x_max"] and
            WORKSPACE["y_min"] < cy < WORKSPACE["y_max"]):
        return "📍 Out of reach", COLOR_OUT_OF_WS
    return "✅ Human can grab this!", COLOR_GRASPABLE


def draw_crosshair(frame, cx, cy, color, size=12, thickness=2):
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)
    cv2.circle(frame, (cx, cy), size // 2, color, thickness)


def draw_info_panel(frame, detections, fps):
    """Draw a semi-transparent HUD panel on the left side."""
    h, w = frame.shape[:2]
    panel_w = 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 20
    def put(text, color=(220, 220, 220), scale=0.48, bold=False):
        nonlocal y
        thickness = 2 if bold else 1
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        y += 20

    put("GRAB DETECTOR 🤖", (0, 220, 120), scale=0.52, bold=True)
    put(f"FPS: {fps:.1f}", (200, 200, 0))
    put(f"Objects seen: {len(detections)}")
    put("─" * 28, (80, 80, 80))

    graspable_count = sum(1 for d in detections if "can grab" in d["decision"])
    put(f"✅ Grabbable: {graspable_count}", COLOR_GRASPABLE)
    put(f"❌ Not grabbable: {len(detections) - graspable_count}", COLOR_NOT_GRASPABLE)
    put("─" * 28, (80, 80, 80))

    for i, d in enumerate(detections[:5]):   # show top 5
        color = d["color"]
        put(f"[{i+1}] {d['label']}", color, bold=True)
        put(f"    Confidence: {d['conf']:.0%}")
        put(f"    Position: ({d['cx']}, {d['cy']})")
        put(f"    {d['decision']}", color)
        put("")


def draw_workspace_boundary(frame):
    """Visualize the robot workspace limits."""
    ws = WORKSPACE
    cv2.rectangle(frame,
                  (ws["x_min"], ws["y_min"]),
                  (ws["x_max"], ws["y_max"]),
                  (100, 100, 255), 1)
    cv2.putText(frame, "WORKSPACE", (ws["x_min"] + 4, ws["y_min"] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run(source=0, conf_threshold=0.45, model_size="n"):
    """
    Main detection loop.

    Args:
        source         : webcam index (int) or video path (str)
        conf_threshold : minimum confidence to display a detection
        model_size     : YOLOv8 variant — n(ano), s(mall), m(edium)
    """
    print(f"[INFO] Loading YOLOv8{model_size} model …")
    model = YOLO(f"yolov8{model_size}.pt")   # auto-downloads on first run
    model.fuse()

    print(f"[INFO] Opening source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolution: {frame_w}×{frame_h}")
    print("[INFO] Press 'q' to quit | 's' to save snapshot | 'p' to pause\n")

    prev_time = time.time()
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended.")
                break

        # ── YOLOv8 inference ──────────────────────────────────────
        results = model(frame, conf=conf_threshold, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            area   = (x2 - x1) * (y2 - y1)
            rx, ry = pixel_to_robot_coords(cx, cy, frame_w, frame_h)
            decision, color = grasp_decision(label, area, cx, cy)

            detections.append({
                "label": label, "conf": conf,
                "cx": cx, "cy": cy, "rx": rx, "ry": ry,
                "area": area, "decision": decision, "color": color,
                "box": (x1, y1, x2, y2)
            })

            # ── Draw bounding box ──────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_crosshair(frame, cx, cy, color)

            # Label tag above box
            tag = f"{label} {conf:.0%} | {decision}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ── HUD overlays ──────────────────────────────────────────
        draw_workspace_boundary(frame)

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        draw_info_panel(frame, detections, fps)

        if paused:
            cv2.putText(frame, "PAUSED", (frame_w // 2 - 50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        cv2.imshow("Industrial Pick & Place Vision System", frame)

        # ── Key handling ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit.")
            break
        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Snapshot saved → {fname}")
        elif key == ord('p'):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Industrial Pick-and-Place Vision System (YOLOv8 + OpenCV)"
    )
    parser.add_argument("--source", default=0,
                        help="Webcam index (0,1,…) or video file path")
    parser.add_argument("--conf",   type=float, default=0.45,
                        help="Detection confidence threshold (default: 0.45)")
    parser.add_argument("--model",  default="n", choices=["n", "s", "m", "l"],
                        help="YOLOv8 model size: n=nano, s=small, m=medium (default: n)")
    args = parser.parse_args()

    # Convert source to int if it's a digit string (webcam index)
    source = int(args.source) if str(args.source).isdigit() else args.source

    run(source=source, conf_threshold=args.conf, model_size=args.model)
