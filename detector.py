"""
GPU-Accelerated CCTV Motion & Event Detection Engine with Mid-Video Crash Recovery.

Leverages NVIDIA RTX 3090 GPU hardware acceleration (OpenCV OpenCL UMat compute
pipeline + asynchronous frame prefetching with grab-based frame skipping) and
supports periodic mid-video checkpointing and exact-frame resume for multi-hour
MP4 recordings.
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.d string."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:04.1f}"


def infer_camera_name(file_path: str) -> str:
    """Infer a friendly camera name from the file name or parent folder."""
    path = Path(file_path)
    stem = path.stem
    parent = path.parent.name

    cam_match = re.search(r"(cam(?:era)?[_\-\s]*\d+|ch(?:annel)?[_\-\s]*\d+)", stem, re.IGNORECASE)
    if cam_match:
        raw = cam_match.group(1).upper().replace("-", "_").replace(" ", "_")
        loc_match = re.search(
            r"(?:cam(?:era)?[_\-\s]*\d+|ch(?:annel)?[_\-\s]*\d+)[_\-\s]+([a-zA-Z]+)",
            stem,
            re.IGNORECASE,
        )
        if loc_match:
            loc = loc_match.group(1)
            return f"{raw} ({loc})"
        return raw

    if re.search(r"(cam|ch\d|camera)", parent, re.IGNORECASE):
        return parent

    parts = re.split(r"[_\-\s]+", stem)
    if len(parts) >= 2:
        return f"{parts[0].title()} {parts[1].title()}"
    return stem


class GPUAccelerator:
    """
    Manages NVIDIA GPU hardware acceleration via OpenCV OpenCL (UMat) and
    pre-compiles GPU kernels to eliminate first-frame compilation latency.
    """

    _initialized: bool = False
    _gpu_available: bool = False
    _device_name: str = "CPU"
    _compute_units: int = 0

    @classmethod
    def initialize(cls, enable_gpu: bool = True) -> Tuple[bool, str]:
        """Initialize and warm up the GPU OpenCL compute pipeline (e.g., NVIDIA GeForce RTX 3090)."""
        if not enable_gpu:
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
            cls._gpu_available = False
            cls._device_name = "CPU (GPU Disabled)"
            return False, cls._device_name

        if cls._initialized and cls._gpu_available:
            cv2.ocl.setUseOpenCL(True)
            return True, cls._device_name

        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.useOpenCL():
                    dev = cv2.ocl.Device.getDefault()
                    name = dev.name().strip() if dev else "OpenCL GPU"
                    cus = int(dev.maxComputeUnits()) if dev else 0
                    cls._device_name = f"{name} ({cus} CUs)" if cus > 0 else name
                    cls._compute_units = cus
                    cls._gpu_available = True

                    # Pre-warm GPU kernels once on a small dummy UMat buffer
                    dummy = cv2.UMat(np.zeros((270, 480, 3), dtype=np.uint8))
                    bg = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=28.0, detectShadows=True)
                    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
                    u_blur = cv2.GaussianBlur(dummy, (5, 5), 0)
                    u_raw = bg.apply(u_blur)
                    _, u_fg = cv2.threshold(u_raw, 200, 255, cv2.THRESH_BINARY)
                    u_diff = cv2.absdiff(u_blur, u_blur)
                    u_gray = cv2.cvtColor(u_diff, cv2.COLOR_BGR2GRAY)
                    _, u_dm = cv2.threshold(u_gray, 12, 255, cv2.THRESH_BINARY)
                    u_dil = cv2.dilate(u_dm, k_close)
                    u_and = cv2.bitwise_and(u_fg, u_dil)
                    u_cln = cv2.morphologyEx(u_and, cv2.MORPH_OPEN, k_open)
                    u_con = cv2.morphologyEx(u_cln, cv2.MORPH_CLOSE, k_close, iterations=2)
                    _ = u_con.get()

                    cls._initialized = True
                    return True, cls._device_name
        except Exception:
            pass

        cls._gpu_available = False
        cls._device_name = "CPU Fallback"
        return False, cls._device_name

    @classmethod
    def device_summary(cls) -> str:
        if not cls._initialized:
            cls.initialize(True)
        return cls._device_name

    @classmethod
    def is_gpu_active(cls) -> bool:
        if not cls._initialized:
            cls.initialize(True)
        return cls._gpu_available and cv2.ocl.useOpenCL()


@dataclass
class DetectionConfig:
    """Configuration parameters for the CCTV event detection pipeline."""

    sample_fps: float = 6.0
    analysis_width: int = 480
    min_area_pct: float = 0.42          # Minimum single-object area (% of frame)
    max_global_light_pct: float = 40.0  # Above this with uniform luminance shift = light variation
    min_solidity: float = 0.42          # Filters out perforated/branch-like fragmented blobs
    min_duration_sec: float = 0.45      # Minimum duration to confirm an event
    merge_gap_sec: float = 1.25         # Merge consecutive detections separated by <= this gap
    pre_roll_sec: float = 2.0           # Seconds before event timestamp for playback start
    sensitivity: str = "Balanced"       # "High", "Balanced", "Strict"
    use_gpu: bool = True                # Use NVIDIA RTX 3090 OpenCL UMat acceleration
    checkpoint_interval_sec: float = 4.0  # Save progress every N seconds during long videos

    def apply_sensitivity_preset(self) -> None:
        """Adjust thresholds based on sensitivity preset."""
        preset = self.sensitivity.lower()
        if preset == "high":
            self.min_area_pct = 0.28
            self.min_solidity = 0.35
            self.min_duration_sec = 0.35
        elif preset == "strict":
            self.min_area_pct = 0.85
            self.min_solidity = 0.50
            self.min_duration_sec = 0.60
        else:  # Balanced
            self.min_area_pct = 0.42
            self.min_solidity = 0.42
            self.min_duration_sec = 0.45


@dataclass
class DetectedEvent:
    """Represents a single verified motion event in a CCTV video file."""

    event_id: int
    file_name: str
    file_path: str
    camera_name: str
    start_time_sec: float
    end_time_sec: float
    peak_time_sec: float
    duration_sec: float
    playback_start_sec: float
    event_type: str
    confidence: float
    max_area_pct: float
    avg_speed_norm: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h) in original video coordinates
    description: str
    displacement_score_pct: float = 0.0      # Cumulative % of video frame pixels affected by the movement
    thumbnail_rgb: Optional[np.ndarray] = field(default=None, repr=False)
    thumbnail_path: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.displacement_score_pct <= 0.0 and self.max_area_pct > 0.0:
            # Fallback estimate if constructed without explicit union mask percentage
            est = self.max_area_pct * (1.0 + self.avg_speed_norm * max(0.5, self.duration_sec) * 2.5)
            self.displacement_score_pct = round(min(100.0, est), 2)

    @property
    def displacement_score_str(self) -> str:
        return f"{self.displacement_score_pct:.2f}%"

    @property
    def start_time_str(self) -> str:
        return format_timestamp(self.start_time_sec)

    @property
    def end_time_str(self) -> str:
        return format_timestamp(self.end_time_sec)

    @property
    def playback_start_str(self) -> str:
        return format_timestamp(self.playback_start_sec)

    @property
    def duration_str(self) -> str:
        return f"{self.duration_sec:.1f}s"

    @property
    def confidence_str(self) -> str:
        return f"{int(round(self.confidence * 100))}%"


@dataclass
class _FrameObservation:
    """
    Lightweight per-frame observation of significant foreground objects.
    Stores no full-resolution video frames to guarantee low, flat RAM usage
    even on 24-hour 4K CCTV streams.
    """

    timestamp_sec: float
    frame_idx: int
    bbox_norm: Tuple[float, float, float, float]  # (x, y, w, h) in [0..1]
    centroid_norm: Tuple[float, float]            # (cx, cy) in [0..1]
    area_pct: float
    aspect_ratio: float                           # w / h
    solidity: float
    fill_ratio: float                             # contour_area / bbox_area


class _AsyncFramePrefetcher:
    """
    Asynchronous producer thread that reads a video file using fast `cap.grab()`
    on non-sampled frames and decodes + resizes only sampled frames in parallel
    with GPU processing.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        start_frame_idx: int,
        step: int,
        proc_w: int,
        proc_h: int,
        thumb_w: int = 320,
        thumb_h: int = 180,
        max_queue_size: int = 24,
    ) -> None:
        self._cap = cap
        self._start_frame_idx = max(0, start_frame_idx)
        self._step = max(1, step)
        self._proc_w = proc_w
        self._proc_h = proc_h
        self._thumb_w = thumb_w
        self._thumb_h = thumb_h
        self._queue: "queue.Queue[Optional[Tuple[int, np.ndarray, np.ndarray]]]" = queue.Queue(
            maxsize=max_queue_size
        )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        frame_idx = self._start_frame_idx
        if frame_idx > 0:
            # Align resume seek to an I-frame at or before frame_idx so decoder state is clean
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
            lookback_f = max(0, frame_idx - int(round(fps * 3.0)))
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, lookback_f)
            cur = lookback_f
            while cur < frame_idx and not self._stop_event.is_set():
                if not self._cap.grab():
                    break
                cur += 1

        try:
            while not self._stop_event.is_set():
                # Fast grab without decoding pixel buffers for skipped frames
                if not self._cap.grab():
                    break

                if (frame_idx - self._start_frame_idx) % self._step != 0:
                    frame_idx += 1
                    continue

                ret, frame = self._cap.retrieve()
                if not ret or frame is None:
                    break

                # Downscale to analysis resolution and compact thumbnail resolution
                small = cv2.resize(
                    frame, (self._proc_w, self._proc_h), interpolation=cv2.INTER_AREA
                )
                if self._proc_w == self._thumb_w and self._proc_h == self._thumb_h:
                    thumb_bgr = small.copy()
                else:
                    thumb_bgr = cv2.resize(
                        small, (self._thumb_w, self._thumb_h), interpolation=cv2.INTER_AREA
                    )

                while not self._stop_event.is_set():
                    try:
                        self._queue.put((frame_idx, small, thumb_bgr), timeout=0.1)
                        break
                    except queue.Full:
                        continue

                frame_idx += 1
        finally:
            while not self._stop_event.is_set():
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass

    def read_next(self) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
                return item
            except queue.Empty:
                if not self._thread.is_alive():
                    return None
        return None

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class VideoEventAnalyzer:
    """
    Analyzes MP4 files for real security/activity events using RTX 3090 GPU
    acceleration while suppressing light variations, moving shadows, and wind
    moving tree branches, and supporting mid-video crash checkpoints.
    """

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()
        self._cancel_requested = False
        self._next_event_id = 1
        self.gpu_active, self.gpu_device_name = GPUAccelerator.initialize(self.config.use_gpu)

    def cancel(self) -> None:
        """Request cancellation of an ongoing folder/file scan."""
        self._cancel_requested = True

    def reset(self, start_event_id: int = 1) -> None:
        """Reset cancellation flag and set next event ID."""
        self._cancel_requested = False
        self._next_event_id = max(1, int(start_event_id))
        self.gpu_active, self.gpu_device_name = GPUAccelerator.initialize(self.config.use_gpu)

    @staticmethod
    def list_mp4_files(folder_path: str) -> List[str]:
        """Return sorted list of .mp4 files in the given folder (and subfolders)."""
        root = Path(folder_path)
        if not root.exists():
            return []
        mp4_files: List[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".mp4":
                mp4_files.append(path)
        mp4_files.sort(key=lambda p: (p.parent.name.lower(), p.name.lower()))
        return [str(p) for p in mp4_files]

    def scan_folder(
        self,
        folder_path: str,
        on_event_detected: Optional[Callable[[DetectedEvent], None]] = None,
        on_progress: Optional[
            Callable[[int, int, str, float, float, float], None]
        ] = None,
        resume_file_offsets: Optional[Dict[str, int]] = None,
        completed_files: Optional[set[str]] = None,
        start_event_id: int = 1,
        on_checkpoint: Optional[
            Callable[[str, int, int, float, float, bool], None]
        ] = None,
    ) -> List[DetectedEvent]:
        """
        Scan all MP4 files in `folder_path`, supporting mid-folder and mid-file resume.

        Callbacks:
          - `on_event_detected(event)`
          - `on_progress(file_idx, total_files, file_name, file_frac, overall_frac, effective_fps)`
          - `on_checkpoint(file_path, last_frame_idx, total_frames, last_time_sec, duration_sec, is_file_done)`
        """
        self.reset(start_event_id=start_event_id)
        self.config.apply_sensitivity_preset()
        files = self.list_mp4_files(folder_path)
        all_events: List[DetectedEvent] = []
        total_files = len(files)
        done_set = {os.path.normcase(os.path.abspath(p)) for p in (completed_files or set())}
        offsets_norm = {
            os.path.normcase(os.path.abspath(k)): int(v)
            for k, v in (resume_file_offsets or {}).items()
        }

        for idx, file_path in enumerate(files):
            if self._cancel_requested:
                break

            norm_path = os.path.normcase(os.path.abspath(file_path))
            file_name = os.path.basename(file_path)

            if norm_path in done_set:
                if on_progress:
                    overall = (idx + 1.0) / max(1, total_files)
                    on_progress(idx + 1, total_files, file_name, 1.0, overall, 0.0)
                continue

            start_frame = offsets_norm.get(norm_path, 0)

            def _file_prog(
                frac: float,
                eff_fps: float = 0.0,
                idx: int = idx,
                fname: str = file_name,
            ) -> None:
                if on_progress:
                    overall = (idx + frac) / max(1, total_files)
                    on_progress(idx + 1, total_files, fname, frac, overall, eff_fps)

            def _file_ckpt(
                f_idx: int,
                tot_f: int,
                t_sec: float,
                dur_sec: float,
                is_done: bool,
                fpath: str = file_path,
            ) -> None:
                if on_checkpoint:
                    on_checkpoint(fpath, f_idx, tot_f, t_sec, dur_sec, is_done)

            events = self.analyze_video_file(
                file_path=file_path,
                on_event_detected=on_event_detected,
                on_file_progress=_file_prog,
                start_frame_idx=start_frame,
                on_checkpoint=_file_ckpt,
            )
            all_events.extend(events)

        return all_events

    def analyze_video_file(
        self,
        file_path: str,
        on_event_detected: Optional[Callable[[DetectedEvent], None]] = None,
        on_file_progress: Optional[Callable[[float, float], None]] = None,
        start_frame_idx: int = 0,
        on_checkpoint: Optional[Callable[[int, int, float, float, bool], None]] = None,
    ) -> List[DetectedEvent]:
        """
        Analyze a single MP4 video file using RTX 3090 GPU OpenCL UMat acceleration
        (when enabled), starting from `start_frame_idx` (for crash recovery on long videos).
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = (total_frames / fps) if total_frames > 0 else 0.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

        step = max(1, int(round(fps / max(1.0, self.config.sample_fps))))
        effective_sample_fps = fps / step

        scale = min(1.0, self.config.analysis_width / float(max(1, orig_w)))
        proc_w = max(64, int(round(orig_w * scale)))
        proc_h = max(48, int(round(orig_h * scale)))
        frame_area = float(proc_w * proc_h)

        # Warmup window: ~0.6s of sampled frames before active detection
        warmup_samples = max(3, int(round(effective_sample_fps * 0.6)))
        actual_seek_frame = max(0, start_frame_idx - warmup_samples * step)

        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=int(max(60, effective_sample_fps * 25)),
            varThreshold=28.0,
            detectShadows=True,
        )

        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

        use_umat = self.gpu_active and self.config.use_gpu

        prev_fg_binary: Optional[np.ndarray] = None
        flicker_ema: Optional[np.ndarray] = None
        prev_mean_luma: Optional[float] = None
        prev_bgr_cpu: Optional[np.ndarray] = None
        prev_blur_umat: Optional[cv2.UMat] = None

        active_observations: List[_FrameObservation] = []
        event_union_mask: Optional[np.ndarray] = None
        peak_thumb_bgr: Optional[np.ndarray] = None
        peak_area_so_far: float = -1.0
        last_motion_time: float = -999.0
        confirmed_events: List[DetectedEvent] = []

        sampled_count = 0
        file_name = os.path.basename(file_path)
        camera_name = infer_camera_name(file_path)

        t_scan_start = time.perf_counter()
        last_ckpt_wall_time = t_scan_start
        last_processed_frame_idx = max(0, start_frame_idx)

        prefetcher = _AsyncFramePrefetcher(
            cap=cap,
            start_frame_idx=actual_seek_frame,
            step=step,
            proc_w=proc_w,
            proc_h=proc_h,
        )

        try:
            while not self._cancel_requested:
                item = prefetcher.read_next()
                if item is None:
                    break

                frame_idx, small, thumb_bgr = item
                last_processed_frame_idx = frame_idx
                timestamp_sec = frame_idx / fps
                sampled_count += 1

                now = time.perf_counter()
                elapsed = max(1e-3, now - t_scan_start)
                effective_video_fps = (sampled_count * step) / elapsed

                if total_frames > 0 and sampled_count % 8 == 0:
                    if on_file_progress:
                        try:
                            on_file_progress(
                                min(0.99, frame_idx / float(total_frames)),
                                effective_video_fps,
                            )
                        except TypeError:
                            # Backwards-compatible 1-arg progress callback
                            on_file_progress(min(0.99, frame_idx / float(total_frames)))  # type: ignore[call-arg]

                # Periodic mid-video checkpoint save (every checkpoint_interval_sec)
                if (
                    on_checkpoint
                    and frame_idx >= start_frame_idx
                    and (now - last_ckpt_wall_time) >= self.config.checkpoint_interval_sec
                ):
                    last_ckpt_wall_time = now
                    on_checkpoint(frame_idx, total_frames, timestamp_sec, duration_sec, False)

                is_warmup = (sampled_count <= warmup_samples) or (frame_idx < start_frame_idx)
                learning_rate = 0.08 if is_warmup else -1.0

                if use_umat:
                    # --- NVIDIA RTX 3090 OpenCL UMat GPU Pipeline ---
                    u_small = cv2.UMat(small)
                    u_blur = cv2.GaussianBlur(u_small, (5, 5), 0)
                    mean_bgr = cv2.mean(u_blur)[:3]
                    mean_luma = 0.114 * mean_bgr[0] + 0.587 * mean_bgr[1] + 0.299 * mean_bgr[2]
                    luma_jump = abs(mean_luma - prev_mean_luma) if prev_mean_luma is not None else 0.0
                    prev_mean_luma = mean_luma

                    if luma_jump > 14.0:
                        bg_sub.apply(u_blur, learningRate=0.25)
                        prev_blur_umat = u_blur
                        continue

                    u_raw = bg_sub.apply(u_blur, learningRate=learning_rate)
                    _, u_fg = cv2.threshold(u_raw, 200, 255, cv2.THRESH_BINARY)

                    if prev_blur_umat is not None:
                        prev_mean = cv2.mean(prev_blur_umat)[:3]
                        shift = (
                            mean_bgr[0] - prev_mean[0],
                            mean_bgr[1] - prev_mean[1],
                            mean_bgr[2] - prev_mean[2],
                            0.0,
                        )
                        u_shifted = cv2.subtract(u_blur, shift)
                        u_diff3 = cv2.absdiff(u_shifted, prev_blur_umat)
                        u_diff_gray = cv2.cvtColor(u_diff3, cv2.COLOR_BGR2GRAY)
                        _, u_diff_mask = cv2.threshold(u_diff_gray, 11, 255, cv2.THRESH_BINARY)
                        u_diff_dil = cv2.dilate(u_diff_mask, close_kernel, iterations=1)
                        u_fg = cv2.bitwise_and(u_fg, u_diff_dil)

                    prev_blur_umat = u_blur

                    if is_warmup:
                        continue

                    u_cleaned = cv2.morphologyEx(u_fg, cv2.MORPH_OPEN, open_kernel, iterations=1)
                    fg_mask = u_fg.get()
                    cleaned = u_cleaned.get()
                else:
                    # --- CPU Pipeline Fallback ---
                    blurred_bgr = cv2.GaussianBlur(small, (5, 5), 0)
                    mean_bgr = cv2.mean(blurred_bgr)[:3]
                    mean_luma = 0.114 * mean_bgr[0] + 0.587 * mean_bgr[1] + 0.299 * mean_bgr[2]
                    luma_jump = abs(mean_luma - prev_mean_luma) if prev_mean_luma is not None else 0.0
                    prev_mean_luma = mean_luma

                    if luma_jump > 14.0:
                        bg_sub.apply(blurred_bgr, learningRate=0.25)
                        prev_bgr_cpu = blurred_bgr
                        continue

                    raw_mask = bg_sub.apply(blurred_bgr, learningRate=learning_rate)
                    _, fg_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

                    if prev_bgr_cpu is not None:
                        prev_mean = cv2.mean(prev_bgr_cpu)[:3]
                        shift = (
                            mean_bgr[0] - prev_mean[0],
                            mean_bgr[1] - prev_mean[1],
                            mean_bgr[2] - prev_mean[2],
                            0.0,
                        )
                        shifted = cv2.subtract(blurred_bgr, shift)
                        diff3 = cv2.absdiff(shifted, prev_bgr_cpu)
                        diff_max = np.max(diff3, axis=2)
                        _, diff_mask = cv2.threshold(diff_max, 14, 255, cv2.THRESH_BINARY)
                        diff_dilated = cv2.dilate(diff_mask, close_kernel, iterations=1)
                        fg_mask = cv2.bitwise_and(fg_mask, diff_dilated)

                    prev_bgr_cpu = blurred_bgr

                    if is_warmup:
                        continue

                    cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

                # Global illumination / camera shake rejection
                raw_fg_pct = (np.count_nonzero(fg_mask) / frame_area) * 100.0
                if raw_fg_pct > self.config.max_global_light_pct:
                    if use_umat and prev_blur_umat is not None:
                        bg_sub.apply(prev_blur_umat, learningRate=0.20)
                    elif prev_bgr_cpu is not None:
                        bg_sub.apply(prev_bgr_cpu, learningRate=0.20)
                    continue

                # Rolling per-pixel flicker EMA to suppress chronic swaying foliage
                fg_unit = (fg_mask > 0).astype(np.float32)
                if prev_fg_binary is not None:
                    toggle = np.abs(fg_unit - prev_fg_binary)
                    if flicker_ema is None:
                        flicker_ema = toggle * 0.1
                    else:
                        flicker_ema = 0.90 * flicker_ema + 0.10 * toggle
                prev_fg_binary = fg_unit

                if flicker_ema is not None:
                    chronic_flicker = (flicker_ema > 0.34).astype(np.uint8) * 255
                    eroded_in_flicker = cv2.erode(cleaned, open_kernel, iterations=1)
                    cleaned = np.where(chronic_flicker > 0, eroded_in_flicker, cleaned)

                if use_umat:
                    u_conn = cv2.morphologyEx(
                        cv2.UMat(cleaned), cv2.MORPH_CLOSE, close_kernel, iterations=2
                    )
                    connected = u_conn.get()
                else:
                    connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=2)

                obs = self._extract_best_observation(
                    connected_mask=connected,
                    raw_cleaned_mask=cleaned,
                    proc_w=proc_w,
                    proc_h=proc_h,
                    timestamp_sec=timestamp_sec,
                    frame_idx=frame_idx,
                )

                if obs is not None:
                    active_observations.append(obs)
                    last_motion_time = timestamp_sec
                    # Accumulate affected pixels for this event in event_union_mask
                    if event_union_mask is None:
                        event_union_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
                    bx_n, by_n, bw_n, bh_n = obs.bbox_norm
                    rx1 = max(0, int(round(bx_n * proc_w)))
                    ry1 = max(0, int(round(by_n * proc_h)))
                    rx2 = min(proc_w, int(round((bx_n + bw_n) * proc_w)))
                    ry2 = min(proc_h, int(round((by_n + bh_n) * proc_h)))
                    if rx2 > rx1 and ry2 > ry1:
                        event_union_mask[ry1:ry2, rx1:rx2] = cv2.bitwise_or(
                            event_union_mask[ry1:ry2, rx1:rx2],
                            connected[ry1:ry2, rx1:rx2],
                        )
                    if obs.area_pct > peak_area_so_far:
                        peak_area_so_far = obs.area_pct
                        peak_thumb_bgr = thumb_bgr
                else:
                    if (
                        active_observations
                        and (timestamp_sec - last_motion_time) >= self.config.merge_gap_sec
                    ):
                        disp_pct = (
                            (float(np.count_nonzero(event_union_mask)) / frame_area) * 100.0
                            if event_union_mask is not None
                            else 0.0
                        )
                        event = self._finalize_event(
                            observations=active_observations,
                            peak_thumb_bgr=peak_thumb_bgr,
                            displacement_score_pct=disp_pct,
                            file_path=file_path,
                            file_name=file_name,
                            camera_name=camera_name,
                            orig_w=orig_w,
                            orig_h=orig_h,
                        )
                        active_observations = []
                        event_union_mask = None
                        peak_thumb_bgr = None
                        peak_area_so_far = -1.0
                        if event is not None:
                            confirmed_events.append(event)
                            if on_event_detected:
                                on_event_detected(event)
                            if on_checkpoint:
                                last_ckpt_wall_time = time.perf_counter()
                                on_checkpoint(
                                    frame_idx, total_frames, timestamp_sec, duration_sec, False
                                )

            # Finalize any event still active when reaching EOF
            if active_observations and not self._cancel_requested:
                disp_pct = (
                    (float(np.count_nonzero(event_union_mask)) / frame_area) * 100.0
                    if event_union_mask is not None
                    else 0.0
                )
                event = self._finalize_event(
                    observations=active_observations,
                    peak_thumb_bgr=peak_thumb_bgr,
                    displacement_score_pct=disp_pct,
                    file_path=file_path,
                    file_name=file_name,
                    camera_name=camera_name,
                    orig_w=orig_w,
                    orig_h=orig_h,
                )
                if event is not None:
                    confirmed_events.append(event)
                    if on_event_detected:
                        on_event_detected(event)
        finally:
            prefetcher.stop()
            cap.release()

        if on_checkpoint:
            is_completed = not self._cancel_requested
            final_frame = total_frames if is_completed else last_processed_frame_idx
            final_time = duration_sec if is_completed else (last_processed_frame_idx / fps)
            on_checkpoint(final_frame, total_frames, final_time, duration_sec, is_completed)

        if on_file_progress and not self._cancel_requested:
            try:
                on_file_progress(1.0, 0.0)
            except TypeError:
                on_file_progress(1.0)  # type: ignore[call-arg]

        return confirmed_events

    def _extract_best_observation(
        self,
        connected_mask: np.ndarray,
        raw_cleaned_mask: np.ndarray,
        proc_w: int,
        proc_h: int,
        timestamp_sec: float,
        frame_idx: int,
    ) -> Optional[_FrameObservation]:
        """Find the most significant coherent object contour in the current frame."""
        contours, _ = cv2.findContours(
            connected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        frame_area = float(proc_w * proc_h)
        best_obs: Optional[_FrameObservation] = None
        best_score = 0.0

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            area_pct = (area / frame_area) * 100.0
            if area_pct < self.config.min_area_pct:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 6 or h < 6:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / max(1.0, hull_area)

            roi_raw = raw_cleaned_mask[y : y + h, x : x + w]
            active_pixels = float(np.count_nonzero(roi_raw))
            fill_ratio = active_pixels / float(w * h)

            if solidity < self.config.min_solidity or fill_ratio < 0.14:
                continue

            cx = (x + 0.5 * w) / float(proc_w)
            cy = (y + 0.5 * h) / float(proc_h)
            bbox_norm = (
                x / float(proc_w),
                y / float(proc_h),
                w / float(proc_w),
                h / float(proc_h),
            )
            aspect_ratio = float(w) / float(max(1, h))

            score = area_pct * solidity * (0.5 + fill_ratio)
            if score > best_score:
                best_score = score
                best_obs = _FrameObservation(
                    timestamp_sec=timestamp_sec,
                    frame_idx=frame_idx,
                    bbox_norm=bbox_norm,
                    centroid_norm=(cx, cy),
                    area_pct=area_pct,
                    aspect_ratio=aspect_ratio,
                    solidity=solidity,
                    fill_ratio=fill_ratio,
                )

        return best_obs

    def _finalize_event(
        self,
        observations: List[_FrameObservation],
        peak_thumb_bgr: Optional[np.ndarray],
        displacement_score_pct: float,
        file_path: str,
        file_name: str,
        camera_name: str,
        orig_w: int,
        orig_h: int,
    ) -> Optional[DetectedEvent]:
        """
        Evaluate a temporal sequence of frame observations to decide if it is a
        real event (vs. wind/oscillation noise) and classify its characteristics.
        """
        if len(observations) < 2:
            return None

        start_t = observations[0].timestamp_sec
        end_t = observations[-1].timestamp_sec
        duration = max(0.1, end_t - start_t)

        if duration < self.config.min_duration_sec:
            return None

        centroids = np.array([o.centroid_norm for o in observations], dtype=np.float64)
        net_disp = float(np.linalg.norm(centroids[-1] - centroids[0]))
        step_deltas = np.diff(centroids, axis=0)
        step_lengths = np.linalg.norm(step_deltas, axis=1)
        total_path = float(np.sum(step_lengths))
        directional_efficiency = net_disp / max(1e-5, total_path)
        avg_speed_norm = net_disp / duration

        areas = np.array([o.area_pct for o in observations], dtype=np.float64)
        max_area_pct = float(np.max(areas))
        mean_solidity = float(np.mean([o.solidity for o in observations]))
        mean_fill = float(np.mean([o.fill_ratio for o in observations]))
        median_aspect = float(np.median([o.aspect_ratio for o in observations]))

        bboxes = np.array([o.bbox_norm for o in observations], dtype=np.float64)
        widths = bboxes[:, 2]
        heights = bboxes[:, 3]
        max_box_span = float(np.max(np.linalg.norm(centroids - centroids[0], axis=1)))

        # Reject wind-blown tree branches that oscillate back and forth in place
        if (
            net_disp < 0.055
            and directional_efficiency < 0.28
            and mean_fill < 0.30
            and mean_solidity < 0.68
        ):
            return None

        if total_path > 0.20 and directional_efficiency < 0.18 and net_disp < 0.07:
            return None

        peak_idx = int(np.argmax(areas))
        peak_obs = observations[peak_idx]
        bx, by, bw, bh = peak_obs.bbox_norm
        orig_bbox = (
            int(round(bx * orig_w)),
            int(round(by * orig_h)),
            int(round(bw * orig_w)),
            int(round(bh * orig_h)),
        )

        event_type, description, confidence = self._classify_event(
            net_disp=net_disp,
            max_box_span=max_box_span,
            directional_efficiency=directional_efficiency,
            avg_speed_norm=avg_speed_norm,
            max_area_pct=max_area_pct,
            median_aspect=median_aspect,
            mean_solidity=mean_solidity,
            mean_fill=mean_fill,
            duration=duration,
            widths=widths,
            heights=heights,
        )

        thumbnail_rgb: Optional[np.ndarray] = None
        if peak_thumb_bgr is not None:
            vis = peak_thumb_bgr.copy()
            th, tw = vis.shape[:2]
            tx = int(round(bx * tw))
            ty = int(round(by * th))
            tw_box = int(round(bw * tw))
            th_box = int(round(bh * th))
            cv2.rectangle(vis, (tx, ty), (tx + tw_box, ty + th_box), (0, 215, 255), 2)
            label = f"{event_type} ({format_timestamp(peak_obs.timestamp_sec)})"
            cv2.rectangle(
                vis,
                (tx, max(0, ty - 18)),
                (min(tw, tx + max(tw_box, 145)), ty),
                (0, 215, 255),
                -1,
            )
            cv2.putText(
                vis,
                label,
                (tx + 3, max(12, ty - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            thumbnail_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        event_id = self._next_event_id
        self._next_event_id += 1
        playback_start = max(0.0, start_t - self.config.pre_roll_sec)
        final_disp_pct = round(max(max_area_pct, min(100.0, displacement_score_pct)), 2)

        return DetectedEvent(
            event_id=event_id,
            file_name=file_name,
            file_path=file_path,
            camera_name=camera_name,
            start_time_sec=round(start_t, 2),
            end_time_sec=round(end_t, 2),
            peak_time_sec=round(peak_obs.timestamp_sec, 2),
            duration_sec=round(duration, 2),
            playback_start_sec=round(playback_start, 2),
            event_type=event_type,
            confidence=round(min(0.99, max(0.55, confidence)), 2),
            max_area_pct=round(max_area_pct, 2),
            avg_speed_norm=round(avg_speed_norm, 3),
            bounding_box=orig_bbox,
            description=description,
            displacement_score_pct=final_disp_pct,
            thumbnail_rgb=thumbnail_rgb,
        )

    @staticmethod
    def _classify_event(
        net_disp: float,
        max_box_span: float,
        directional_efficiency: float,
        avg_speed_norm: float,
        max_area_pct: float,
        median_aspect: float,
        mean_solidity: float,
        mean_fill: float,
        duration: float,
        widths: np.ndarray,
        heights: np.ndarray,
    ) -> Tuple[str, str, float]:
        if (
            median_aspect >= 1.20
            and (net_disp >= 0.14 or avg_speed_norm >= 0.08)
            and directional_efficiency >= 0.45
            and max_area_pct >= 1.5
        ):
            conf = 0.78 + min(0.18, directional_efficiency * 0.15 + (max_area_pct / 60.0))
            desc = (
                f"Vehicle/car moving across frame (speed {avg_speed_norm * 100:.0f}%/s, "
                f"area {max_area_pct:.1f}%, duration {duration:.1f}s)"
            )
            return "Car Passing", desc, conf

        if (
            net_disp < 0.16
            and max_box_span < 0.18
            and median_aspect <= 1.15
            and mean_solidity >= 0.62
            and mean_fill >= 0.30
        ):
            conf = 0.76 + min(0.18, mean_solidity * 0.18)
            desc = (
                f"Localized door/gate movement (stationary zone, solidity {mean_solidity:.2f}, "
                f"area {max_area_pct:.1f}%, duration {duration:.1f}s)"
            )
            return "Door Opening", desc, conf

        if (
            median_aspect < 1.20
            and net_disp >= 0.10
            and directional_efficiency >= 0.35
        ):
            conf = 0.77 + min(0.19, directional_efficiency * 0.18)
            desc = (
                f"Person walking across scene (displacement {net_disp * 100:.0f}% of frame, "
                f"area {max_area_pct:.1f}%, duration {duration:.1f}s)"
            )
            return "Person Walking", desc, conf

        conf = 0.68 + min(0.20, directional_efficiency * 0.15 + mean_solidity * 0.10)
        desc = (
            f"Coherent motion detected (area {max_area_pct:.1f}%, "
            f"displacement {net_disp * 100:.0f}%, duration {duration:.1f}s)"
        )
        return "Movement Detected", desc, conf
