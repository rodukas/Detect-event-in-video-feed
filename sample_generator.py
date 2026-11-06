"""
Synthetic 4-Camera CCTV Video Feed Generator.

Generates realistic multi-camera MP4 test feeds containing:
  - Continuous wind-blown tree branches/foliage (should be ignored)
  - Gradual & sudden ambient light variations / clouds passing (should be ignored)
  - Genuine events:
      * Cam 1 (Front Door): Door opening at ~5.0s, Person walking at ~11.0s
      * Cam 2 (Driveway): Car passing at ~6.0s
      * Cam 3 (Backyard): Only wind & light variations (0 events expected)
      * Cam 4 (Side Gate): Person walking at ~7.0s
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np


def _draw_house_background(width: int, height: int, cam_id: int) -> np.ndarray:
    """Create a static base background for a given CCTV camera view."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky / upper wall gradient
    for y in range(height // 3):
        val = int(110 + (y / (height / 3)) * 30)
        bg[y, :] = (val + 25, val + 15, val)

    # Lawn / ground
    bg[height // 3 :, :] = (65, 95, 70)

    if cam_id == 1:  # Front Door & Walkway
        # House wall
        cv2.rectangle(bg, (40, 30), (320, 250), (145, 155, 165), -1)
        # Closed door frame
        cv2.rectangle(bg, (140, 110), (210, 250), (60, 75, 110), -1)
        cv2.rectangle(bg, (140, 110), (210, 250), (40, 50, 70), 2)
        # Walkway
        pts = np.array([[140, 250], [210, 250], [290, height], [90, height]], np.int32)
        cv2.fillPoly(bg, [pts], (130, 130, 130))
    elif cam_id == 2:  # Driveway / Street
        # Asphalt road/driveway across middle-bottom
        cv2.rectangle(bg, (0, 175), (width, 295), (82, 82, 86), -1)
        # Dashed road line
        for x in range(20, width, 60):
            cv2.rectangle(bg, (x, 232), (x + 30, 238), (160, 165, 170), -1)
    elif cam_id == 3:  # Backyard (trees + lawn only)
        # Patio stones
        cv2.rectangle(bg, (60, 210), (380, 310), (120, 125, 128), -1)
    else:  # Cam 4: Side Gate & Path
        # Fence
        for x in range(40, width - 40, 28):
            cv2.rectangle(bg, (x, 90), (x + 20, 210), (95, 115, 135), -1)
        # Side path
        cv2.rectangle(bg, (0, 235), (width, 305), (125, 125, 128), -1)

    # Trunk of a tree on the right side
    cv2.rectangle(bg, (width - 115, 90), (width - 95, 235), (50, 70, 95), -1)
    return bg


def _overlay_wind_and_light(
    frame: np.ndarray, t: float, width: int, height: int
) -> np.ndarray:
    """
    Simulate:
      1. Swaying tree branches/leaves in the wind (high-frequency oscillation of small leaves).
      2. Gradual sunlight fluctuations + a sudden cloud/auto-exposure jump around t=2.5s.
    """
    out = frame.copy()

    # Swaying leaves cluster near top-right tree (x ~ width-150..width-40, y ~ 35..140)
    rng = np.random.RandomState(42)
    num_leaves = 28
    base_xs = rng.randint(width - 155, width - 45, size=num_leaves)
    base_ys = rng.randint(35, 135, size=num_leaves)
    phases = rng.uniform(0, 2 * math.pi, size=num_leaves)

    for i in range(num_leaves):
        # High-frequency back-and-forth sway in the wind
        dx = int(round(5.0 * math.sin(t * 7.5 + phases[i])))
        dy = int(round(3.0 * math.cos(t * 9.0 + phases[i])))
        lx = int(base_xs[i] + dx)
        ly = int(base_ys[i] + dy)
        cv2.circle(out, (lx, ly), 3, (45, 125, 55), -1)

    # Gradual brightness wave (+- 5 intensity units)
    gradual_shift = 5.0 * math.sin(t * 0.9)

    # Sudden global ambient light shift between t=2.3s and t=2.8s (e.g., cloud / floodlight)
    sudden_shift = 22.0 if (2.3 <= t <= 2.6) else 0.0

    total_shift = gradual_shift + sudden_shift
    if abs(total_shift) > 0.5:
        out = np.clip(out.astype(np.float32) + total_shift, 0, 255).astype(np.uint8)

    return out


def _draw_door_opening(frame: np.ndarray, t: float, start_t: float, end_t: float) -> None:
    """Draw a solid front door swinging open and revealing a bright interior."""
    if not (start_t <= t <= end_t):
        return
    progress = (t - start_t) / max(0.01, end_t - start_t)
    # Swing open during first half, stay open / close during second half
    open_frac = math.sin(min(1.0, progress) * math.pi)
    open_w = int(round(68 * max(0.25, open_frac)))
    # Interior lit doorway
    cv2.rectangle(frame, (140, 110), (140 + open_w, 250), (215, 225, 240), -1)
    # Swinging door leaf
    door_x = 140 + open_w
    cv2.rectangle(frame, (door_x, 112), ( min(235, door_x + 22), 248), (35, 45, 75), -1)


def _draw_person_walking(
    frame: np.ndarray,
    t: float,
    start_t: float,
    end_t: float,
    start_pos: tuple[int, int],
    end_pos: tuple[int, int],
) -> None:
    """Draw a solid human silhouette walking steadily from start_pos to end_pos."""
    if not (start_t <= t <= end_t):
        return
    alpha = (t - start_t) / max(0.01, end_t - start_t)
    cx = int(round(start_pos[0] + alpha * (end_pos[0] - start_pos[0])))
    cy = int(round(start_pos[1] + alpha * (end_pos[1] - start_pos[1])))

    # Head
    cv2.circle(frame, (cx, cy - 38), 11, (210, 190, 175), -1)
    # Torso (solid high-contrast jacket)
    cv2.rectangle(frame, (cx - 16, cy - 26), (cx + 16, cy + 16), (220, 65, 45), -1)
    # Legs with slight stride animation
    stride = int(round(7 * math.sin(t * 8.0)))
    cv2.rectangle(frame, (cx - 14, cy + 16), (cx - 2 + stride // 2, cy + 48), (40, 45, 60), -1)
    cv2.rectangle(frame, (cx + 2 - stride // 2, cy + 16), (cx + 14, cy + 48), (35, 40, 55), -1)


def _draw_car_passing(
    frame: np.ndarray,
    t: float,
    start_t: float,
    end_t: float,
    y_top: int,
    width: int,
) -> None:
    """Draw a car driving horizontally across the driveway/road."""
    if not (start_t <= t <= end_t):
        return
    alpha = (t - start_t) / max(0.01, end_t - start_t)
    car_w = 135
    car_h = 58
    x = int(round(-car_w + alpha * (width + car_w * 0.8)))
    y = y_top

    x1 = max(0, x)
    x2 = min(width, x + car_w)
    if x2 <= x1:
        return

    # Main car body
    cv2.rectangle(frame, (x1, y + 16), (x2, y + car_h), (35, 185, 235), -1)
    # Cabin / roof
    cab_x1 = max(0, x + 24)
    cab_x2 = min(width, x + car_w - 28)
    if cab_x2 > cab_x1:
        cv2.rectangle(frame, (cab_x1, y), (cab_x2, y + 20), (190, 220, 240), -1)
    # Wheels
    for wx in (x + 28, x + car_w - 28):
        if 0 <= wx < width:
            cv2.circle(frame, (wx, y + car_h), 12, (25, 25, 25), -1)


def generate_sample_cctv_folder(
    output_dir: str,
    duration_sec: float = 15.0,
    fps: float = 20.0,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[str]:
    """
    Generate 4 CCTV camera MP4 files in `output_dir` and return their paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    width, height = 640, 360
    total_frames = int(round(duration_sec * fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    specs = [
        ("Cam1_FrontDoor_2026-11-05.mp4", 1),
        ("Cam2_Driveway_2026-11-05.mp4", 2),
        ("Cam3_Backyard_2026-11-05.mp4", 3),
        ("Cam4_SideGate_2026-11-05.mp4", 4),
    ]

    created_files: List[str] = []

    for idx, (fname, cam_id) in enumerate(specs):
        if on_progress:
            on_progress(idx + 1, len(specs), fname)

        fpath = str(out_path / fname)
        writer = cv2.VideoWriter(fpath, fourcc, fps, (width, height))
        base_bg = _draw_house_background(width, height, cam_id)

        for f_idx in range(total_frames):
            t = f_idx / fps
            frame = _overlay_wind_and_light(base_bg, t, width, height)

            if cam_id == 1:
                # Door opens at 4.8s..6.8s, Person walks up at 10.2s..13.6s
                _draw_door_opening(frame, t, 4.8, 6.8)
                _draw_person_walking(frame, t, 10.2, 13.6, (95, 290), (430, 260))
            elif cam_id == 2:
                # Car passes driveway at 5.5s..8.5s
                _draw_car_passing(frame, t, 5.5, 8.5, y_top=190, width=width)
            elif cam_id == 3:
                # Only wind in trees & light variations -> 0 events should be triggered
                pass
            elif cam_id == 4:
                # Person walks across side gate path at 6.5s..10.2s
                _draw_person_walking(frame, t, 6.5, 10.2, (520, 245), (110, 245))

            # Subtle CCTV OSD timestamp in top-left corner (tiny static box)
            writer.write(frame)

        writer.release()
        created_files.append(fpath)

    return created_files

