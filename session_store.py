"""
Crash-Safe Periodic Session Persistence & Resume Manager.

Regularly saves screening progress (completed files, exact mid-file frame index
and timestamp for long videos, and all detected events with disk-backed thumbnails)
using atomic file replacement so that if the PC crashes or the app is closed,
the user can immediately view previous findings on startup and resume from the
exact point of the last save.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from detector import DetectedEvent, format_timestamp


@dataclass
class FileProgressState:
    """Tracks exact screening progress for an individual MP4 file."""

    file_path: str
    file_name: str
    status: str = "pending"  # "pending", "in_progress", "completed"
    last_frame_idx: int = 0
    total_frames: int = 0
    last_timestamp_sec: float = 0.0
    duration_sec: float = 0.0

    @property
    def fraction_complete(self) -> float:
        if self.status == "completed":
            return 1.0
        if self.total_frames > 0:
            return min(0.99, max(0.0, self.last_frame_idx / float(self.total_frames)))
        if self.duration_sec > 0:
            return min(0.99, max(0.0, self.last_timestamp_sec / float(self.duration_sec)))
        return 0.0


@dataclass
class SessionCheckpoint:
    """Complete serializable state of a screening session."""

    folder_path: str
    status: str = "in_progress"  # "in_progress", "paused", "completed"
    started_at: str = ""
    last_saved_at: str = ""
    gpu_device: str = ""
    sensitivity: str = "Balanced"
    sample_fps: float = 6.0
    use_gpu: bool = True
    next_event_id: int = 1
    files: Dict[str, FileProgressState] = field(default_factory=dict)
    events: List[DetectedEvent] = field(default_factory=list)

    @property
    def is_resumable(self) -> bool:
        """Return True if this session has unfinished work that can be resumed."""
        if not self.folder_path or not os.path.isdir(self.folder_path):
            return False
        if self.status in ("in_progress", "paused"):
            return True
        return any(f.status != "completed" for f in self.files.values())

    @property
    def overall_progress_pct(self) -> float:
        if not self.files:
            return 0.0
        total = len(self.files)
        done_sum = sum(f.fraction_complete for f in self.files.values())
        return round((done_sum / float(total)) * 100.0, 1)

    def resume_summary_text(self) -> str:
        """Human-readable summary of where the session was saved."""
        completed_count = sum(1 for f in self.files.values() if f.status == "completed")
        total_count = len(self.files)
        in_prog = next(
            (f for f in self.files.values() if f.status == "in_progress" and f.last_frame_idx > 0),
            None,
        )
        if in_prog is not None:
            return (
                f"Saved at {self.last_saved_at}  •  {len(self.events)} event(s) found  •  "
                f"File {completed_count + 1}/{total_count}: '{in_prog.file_name}' "
                f"at {format_timestamp(in_prog.last_timestamp_sec)} ({self.overall_progress_pct:.1f}% overall)"
            )
        return (
            f"Saved at {self.last_saved_at}  •  {len(self.events)} event(s) found  •  "
            f"{completed_count}/{total_count} files completed ({self.overall_progress_pct:.1f}%)"
        )


class SessionStore:
    """Manages atomic saving and loading of screening checkpoints and thumbnails."""

    def __init__(self, base_cache_dir: Optional[str] = None) -> None:
        root = Path(base_cache_dir or os.path.join(os.getcwd(), ".cctv_session_cache"))
        self.cache_dir = root
        self.thumbs_dir = root / "thumbnails"
        self.session_file = root / "last_session.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)

    def save_thumbnail(self, event_id: int, thumbnail_rgb: Optional[np.ndarray]) -> Optional[str]:
        """Persist a 320x180 RGB thumbnail to disk as JPEG and return its path."""
        if thumbnail_rgb is None:
            return None
        try:
            self.thumbs_dir.mkdir(parents=True, exist_ok=True)
            thumb_path = self.thumbs_dir / f"event_{event_id:05d}.jpg"
            bgr = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(thumb_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            return str(thumb_path)
        except Exception:
            return None

    @staticmethod
    def load_thumbnail_rgb(thumb_path: Optional[str]) -> Optional[np.ndarray]:
        """Load an RGB thumbnail from disk on demand."""
        if not thumb_path or not os.path.isfile(thumb_path):
            return None
        try:
            bgr = cv2.imread(thumb_path, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def clear_thumbnails(self) -> None:
        """Remove old cached thumbnails when starting a brand-new scan from scratch."""
        if not self.thumbs_dir.exists():
            return
        for p in self.thumbs_dir.glob("event_*.jpg"):
            try:
                p.unlink()
            except OSError:
                pass

    def save_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        """
        Atomically write `checkpoint` to `.cctv_session_cache/last_session.json`
        (and a mirror inside the target folder if writable) so a sudden system
        crash can never corrupt the checkpoint file.
        """
        checkpoint.last_saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not checkpoint.started_at:
            checkpoint.started_at = checkpoint.last_saved_at

        serialized_events: List[Dict[str, Any]] = []
        for ev in checkpoint.events:
            thumb_path = getattr(ev, "thumbnail_path", None)
            if not thumb_path and ev.thumbnail_rgb is not None:
                thumb_path = self.save_thumbnail(ev.event_id, ev.thumbnail_rgb)
                ev.thumbnail_path = thumb_path

            serialized_events.append(
                {
                    "event_id": ev.event_id,
                    "file_name": ev.file_name,
                    "file_path": ev.file_path,
                    "camera_name": ev.camera_name,
                    "start_time_sec": ev.start_time_sec,
                    "end_time_sec": ev.end_time_sec,
                    "peak_time_sec": ev.peak_time_sec,
                    "duration_sec": ev.duration_sec,
                    "playback_start_sec": ev.playback_start_sec,
                    "event_type": ev.event_type,
                    "confidence": ev.confidence,
                    "max_area_pct": ev.max_area_pct,
                    "displacement_score_pct": ev.displacement_score_pct,
                    "avg_speed_norm": ev.avg_speed_norm,
                    "bounding_box": list(ev.bounding_box),
                    "description": ev.description,
                    "thumbnail_path": thumb_path,
                }
            )

        payload = {
            "version": 2,
            "folder_path": checkpoint.folder_path,
            "status": checkpoint.status,
            "started_at": checkpoint.started_at,
            "last_saved_at": checkpoint.last_saved_at,
            "gpu_device": checkpoint.gpu_device,
            "sensitivity": checkpoint.sensitivity,
            "sample_fps": checkpoint.sample_fps,
            "use_gpu": checkpoint.use_gpu,
            "next_event_id": checkpoint.next_event_id,
            "files": {k: asdict(v) for k, v in checkpoint.files.items()},
            "events": serialized_events,
        }

        self._atomic_write_json(self.session_file, payload)

        # Also save a folder-local checkpoint so switching folders preserves per-folder progress
        if checkpoint.folder_path and os.path.isdir(checkpoint.folder_path):
            folder_ckpt = Path(checkpoint.folder_path) / ".cctv_session_checkpoint.json"
            try:
                self._atomic_write_json(folder_ckpt, payload)
            except Exception:
                pass

    @staticmethod
    def _atomic_write_json(target_path: Path, data: Dict[str, Any]) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)

    def load_last_session(self, folder_path: Optional[str] = None) -> Optional[SessionCheckpoint]:
        """
        Load the most recent session checkpoint (either for a specific `folder_path`
        or the global `last_session.json`).
        """
        candidate_paths: List[Path] = []
        if folder_path and os.path.isdir(folder_path):
            candidate_paths.append(Path(folder_path) / ".cctv_session_checkpoint.json")
        candidate_paths.append(self.session_file)

        for path in candidate_paths:
            if not path.is_file():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return self._deserialize_checkpoint(raw)
            except Exception:
                continue
        return None

    def _deserialize_checkpoint(self, raw: Dict[str, Any]) -> SessionCheckpoint:
        files_map: Dict[str, FileProgressState] = {}
        for k, v in raw.get("files", {}).items():
            files_map[k] = FileProgressState(
                file_path=v.get("file_path", k),
                file_name=v.get("file_name", os.path.basename(k)),
                status=v.get("status", "pending"),
                last_frame_idx=int(v.get("last_frame_idx", 0)),
                total_frames=int(v.get("total_frames", 0)),
                last_timestamp_sec=float(v.get("last_timestamp_sec", 0.0)),
                duration_sec=float(v.get("duration_sec", 0.0)),
            )

        events_list: List[DetectedEvent] = []
        for ev_raw in raw.get("events", []):
            thumb_path = ev_raw.get("thumbnail_path")
            bbox_raw = ev_raw.get("bounding_box", [0, 0, 10, 10])
            bbox = (
                int(bbox_raw[0]),
                int(bbox_raw[1]),
                int(bbox_raw[2]),
                int(bbox_raw[3]),
            )
            ev = DetectedEvent(
                event_id=int(ev_raw["event_id"]),
                file_name=str(ev_raw["file_name"]),
                file_path=str(ev_raw["file_path"]),
                camera_name=str(ev_raw.get("camera_name", ev_raw["file_name"])),
                start_time_sec=float(ev_raw["start_time_sec"]),
                end_time_sec=float(ev_raw["end_time_sec"]),
                peak_time_sec=float(ev_raw.get("peak_time_sec", ev_raw["start_time_sec"])),
                duration_sec=float(ev_raw["duration_sec"]),
                playback_start_sec=float(ev_raw["playback_start_sec"]),
                event_type=str(ev_raw["event_type"]),
                confidence=float(ev_raw["confidence"]),
                max_area_pct=float(ev_raw["max_area_pct"]),
                avg_speed_norm=float(ev_raw.get("avg_speed_norm", 0.0)),
                bounding_box=bbox,
                description=str(ev_raw["description"]),
                displacement_score_pct=float(ev_raw.get("displacement_score_pct", 0.0)),
                thumbnail_rgb=self.load_thumbnail_rgb(thumb_path),
                thumbnail_path=thumb_path,
            )
            events_list.append(ev)

        return SessionCheckpoint(
            folder_path=str(raw.get("folder_path", "")),
            status=str(raw.get("status", "in_progress")),
            started_at=str(raw.get("started_at", "")),
            last_saved_at=str(raw.get("last_saved_at", "")),
            gpu_device=str(raw.get("gpu_device", "")),
            sensitivity=str(raw.get("sensitivity", "Balanced")),
            sample_fps=float(raw.get("sample_fps", 6.0)),
            use_gpu=bool(raw.get("use_gpu", True)),
            next_event_id=int(raw.get("next_event_id", len(events_list) + 1)),
            files=files_map,
            events=events_list,
        )

