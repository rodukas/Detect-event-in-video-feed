# Home CCTV 4-Camera Event Detector & Video Reviewer

A desktop application for screening up to 24 hours of MP4 video recordings from 4 home CCTV cameras. It leverages **NVIDIA GeForce RTX 3090 GPU acceleration** (`cv2.UMat` OpenCL pipeline + asynchronous frame prefetching), includes **crash-safe periodic session saving & mid-video resume**, and plays any selected event in an **embedded right-side video frame starting at an I-frame no later than 2 seconds prior to the event**.

## Key Features

- **🎬 Embedded Right-Side Video Player with I-Frame Pre-Roll**:
  - Dedicated video player frame on the right side of the main interface—scroll through the event list using the `Up`/`Down` arrow keys or mouse clicks without any popup windows covering the table.
  - **Artifact-Free I-Frame Seeking**: Whenever a video is played or scrubbed, [`seek_to_iframe_at_or_before`](file:///d:/Vibe%20coding/Detect%20event%20in%20video%20feed/player_window.py#L22-L73) seeks to the nearest preceding **I-frame (keyframe)** (`CAP_PROP_FRAME_TYPE == ord('I')`) whose timestamp is **no later than `event_time - 2.0s`**, ensuring 100% scene visibility from the very first frame without grey P-frame delta compression artifacts.
- **⚡ NVIDIA RTX 3090 GPU Hardware Acceleration**:
  - Automatically detects and pre-warms the **NVIDIA GeForce RTX 3090 (82 Compute Units)** via OpenCV OpenCL (`cv2.UMat`).
  - Uses an asynchronous frame-prefetching producer thread (`cap.grab()` frame skipping + parallel downscaling) feeding the GPU background subtraction, shadow removal, multi-channel frame differencing, and morphological kernels.
- **💾 Crash-Safe Auto-Save & Mid-Video Resume**:
  - Automatically saves checkpoints every **4 seconds**, whenever a new event is detected, whenever a video file completes, and when pausing/closing.
  - On startup, automatically restores and displays all findings from the last session in the table and displays a **Resume Banner** allowing you to continue from the **exact file and mid-video frame/timestamp** of the last save.
- **Environmental Noise Rejection**:
  - Ignores gradual/sudden light variations, moving cloud shadows, and wind moving tree branches/foliage.

## Quick Start

```powershell
.\.venv\Scripts\python.exe app.py
```

Or generate and screen the 4-camera sample feeds:

```powershell
.\.venv\Scripts\python.exe app.py --generate-demo
```

## Running Automated Tests

```powershell
.\.venv\Scripts\pytest -v
```
