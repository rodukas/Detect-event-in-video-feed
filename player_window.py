"""
Embedded Resizable Video Player Frame with I-Frame (Keyframe) Pre-Roll Seeking.

Renders the video associated with the selected `DetectedEvent` directly inside a
large, freely resizable pane (supporting both horizontal and vertical splitter
allocation) and always starts playback at the nearest preceding **I-frame**
(keyframe) no later than `(event.start_time_sec - 2.0)` seconds so the full scene
is crystal clear from the first frame without grey P-frame compression artifacts.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from detector import DetectedEvent, format_timestamp


def set_btn_enabled(
    btn: tk.Button,
    enabled: bool,
    normal_bg: str,
    normal_fg: str = "#ffffff",
    disabled_bg: str = "#1e293b",
    disabled_fg: str = "#94a3b8",
) -> None:
    """
    Set a Tkinter button's state and update its background/foreground colors so
    disabled buttons never suffer from unreadable grey-on-color contrast on Windows.
    """
    if enabled:
        btn.configure(
            state=tk.NORMAL,
            bg=normal_bg,
            fg=normal_fg,
            disabledforeground=disabled_fg,
            cursor="hand2",
        )
    else:
        btn.configure(
            state=tk.DISABLED,
            bg=disabled_bg,
            fg=disabled_fg,
            disabledforeground=disabled_fg,
            cursor="arrow",
        )


def seek_to_iframe_at_or_before(
    cap: cv2.VideoCapture,
    max_target_sec: float,
    fps: float,
    total_frames: int,
) -> int:
    """
    Seek `cap` to the latest I-frame (keyframe, `CAP_PROP_FRAME_TYPE == ord('I')`)
    whose frame index is <= `int(round(max_target_sec * fps))`.

    Guarantees that:
      1. Playback starts no later than `max_target_sec` (e.g. `event_time - 2.0s`).
      2. The decoder starts on a complete reference I-frame, eliminating grey/blocky
         inter-frame (P/B-frame) artifacts on CCTV H.264/H.265/MPEG-4 videos.

    Returns the frame index of the selected I-frame.
    """
    if total_frames <= 0 or fps <= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return 0

    max_frame = max(0, min(total_frames - 1, int(round(max_target_sec * fps))))
    if max_frame == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return 0

    for lookback_sec in (2.5, 6.0, 12.0, 25.0, 60.0):
        start_f = max(0, int(round((max_target_sec - lookback_sec) * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        best_iframe: Optional[int] = None
        steps = max_frame - start_f + 1
        for _ in range(steps):
            if not cap.grab():
                break
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if pos < 0:
                pos = 0
            if pos > max_frame:
                break
            frame_type = int(cap.get(cv2.CAP_PROP_FRAME_TYPE))
            if frame_type == ord("I") or pos == 0:
                best_iframe = pos

        if best_iframe is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_iframe)
            return best_iframe

        if start_f == 0:
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return 0


class EmbeddedVideoPlayerFrame(tk.Frame):
    """
    Large, dynamically resizable embedded video player and event inspector panel.
    Automatically scales the video stream to fill available horizontal & vertical
    space when the user resizes the window or drags the splitter sash.
    """

    def __init__(self, master: tk.Misc, width: int = 800) -> None:
        super().__init__(master, bg="#1e293b", width=width, padx=12, pady=10)
        self.event: Optional[DetectedEvent] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._current_file_path: Optional[str] = None
        self._fps: float = 25.0
        self._total_frames: int = 0
        self._duration_sec: float = 0.0
        self._current_frame_idx: int = 0
        self._iframe_start_idx: int = 0
        self._iframe_start_sec: float = 0.0
        self._last_frame_type_char: str = "I"
        self._last_raw_frame: Optional[np.ndarray] = None
        self._is_playing: bool = False
        self._user_dragging_slider: bool = False
        self._speed_multiplier: float = 1.0
        self._show_bbox_var = tk.BooleanVar(value=True)
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._after_id: Optional[str] = None
        self._resize_after_id: Optional[str] = None

        self._build_ui()
        self.bind("<Destroy>", self._on_widget_destroy)

    def _on_widget_destroy(self, evt: tk.Event) -> None:
        if evt.widget is self:
            self.release()

    def _build_ui(self) -> None:
        # 1. Compact Header bar with Event title & I-Frame Pre-roll status
        header = tk.Frame(self, bg="#0f172a", padx=12, pady=7)
        header.pack(fill=tk.X, side=tk.TOP, pady=(0, 6))

        title_row = tk.Frame(header, bg="#0f172a")
        title_row.pack(fill=tk.X)

        self.title_lbl = tk.Label(
            title_row,
            text="🎬 Video Player — No Event Selected",
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff",
            bg="#0f172a",
            anchor="w",
        )
        self.title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.iframe_badge_lbl = tk.Label(
            title_row,
            text="I-Frame Sync Ready",
            font=("Segoe UI", 8, "bold"),
            fg="#052e16",
            bg="#34d399",
            padx=8,
            pady=2,
        )
        self.iframe_badge_lbl.pack(side=tk.RIGHT)

        self.sub_lbl = tk.Label(
            header,
            text="Select any event in the list to play the video here (starts on an I-frame ≤ Event - 2s). Drag the splitter bar to resize.",
            font=("Segoe UI", 9),
            fg="#7dd3fc",
            bg="#0f172a",
            anchor="w",
            justify=tk.LEFT,
        )
        self.sub_lbl.pack(fill=tk.X, pady=(3, 0))

        # 2. Event Characteristics Summary Card + Transport Controls packed at BOTTOM
        #    so the Video Viewport in the middle gets 100% of remaining vertical/horizontal growth!
        details_box = tk.Frame(self, bg="#0f172a", padx=12, pady=8)
        details_box.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))

        tk.Label(
            details_box,
            text="Selected Event Characteristics",
            font=("Segoe UI", 9, "bold"),
            fg="#7dd3fc",
            bg="#0f172a",
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 2))

        self.info_text = tk.Label(
            details_box,
            text="No event selected.",
            font=("Segoe UI", 9),
            fg="#f1f5f9",
            bg="#0f172a",
            justify=tk.LEFT,
            anchor="nw",
        )
        self.info_text.pack(fill=tk.X)

        # 3. Timeline & Transport Controls (above details_box at bottom)
        controls_box = tk.Frame(self, bg="#0f172a", padx=10, pady=8)
        controls_box.pack(fill=tk.X, side=tk.BOTTOM)

        slider_row = tk.Frame(controls_box, bg="#0f172a")
        slider_row.pack(fill=tk.X, pady=(0, 6))

        self.time_current_lbl = tk.Label(
            slider_row,
            text="00:00:00.0",
            font=("Consolas", 10, "bold"),
            fg="#ffffff",
            bg="#0f172a",
            width=10,
        )
        self.time_current_lbl.pack(side=tk.LEFT)

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_slider = ttk.Scale(
            slider_row,
            from_=0.0,
            to=100.0,
            orient=tk.HORIZONTAL,
            variable=self.seek_var,
            command=self._on_slider_move,
        )
        self.seek_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.seek_slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.seek_slider.bind("<ButtonRelease-1>", self._on_slider_release)

        self.time_total_lbl = tk.Label(
            slider_row,
            text="00:00:00.0",
            font=("Consolas", 10, "bold"),
            fg="#cbd5e1",
            bg="#0f172a",
            width=10,
        )
        self.time_total_lbl.pack(side=tk.RIGHT)

        # Transport buttons row 1
        btn_row1 = tk.Frame(controls_box, bg="#0f172a")
        btn_row1.pack(fill=tk.X, pady=(0, 5))

        self.play_btn = tk.Button(
            btn_row1,
            text="▶ Play",
            font=("Segoe UI", 9, "bold"),
            activebackground="#1e40af",
            activeforeground="#ffffff",
            relief=tk.RAISED,
            bd=1,
            padx=12,
            pady=4,
            command=self.toggle_play_pause,
        )
        self.play_btn.pack(side=tk.LEFT, padx=(0, 6))
        set_btn_enabled(self.play_btn, False, normal_bg="#1d4ed8", normal_fg="#ffffff")

        self.preroll_btn = tk.Button(
            btn_row1,
            text="⏮ I-Frame (≤ Event - 2s)",
            font=("Segoe UI", 9, "bold"),
            activebackground="#0284c7",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=10,
            pady=4,
            command=self.seek_to_event_minus_2s,
        )
        self.preroll_btn.pack(side=tk.LEFT, padx=3)
        set_btn_enabled(self.preroll_btn, False, normal_bg="#38bdf8", normal_fg="#090d16")

        self.exact_btn = tk.Button(
            btn_row1,
            text="🎯 Event Start",
            font=("Segoe UI", 9, "bold"),
            activebackground="#cbd5e1",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=9,
            pady=4,
            command=self._seek_to_exact_event_iframe,
        )
        self.exact_btn.pack(side=tk.LEFT, padx=3)
        set_btn_enabled(self.exact_btn, False, normal_bg="#e2e8f0", normal_fg="#0f172a")

        self.back5_btn = tk.Button(
            btn_row1,
            text="-5s",
            font=("Segoe UI", 9, "bold"),
            activebackground="#cbd5e1",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=8,
            pady=4,
            command=lambda: self.seek_relative(-5.0),
        )
        self.back5_btn.pack(side=tk.LEFT, padx=2)
        set_btn_enabled(self.back5_btn, False, normal_bg="#e2e8f0", normal_fg="#0f172a")

        self.fwd5_btn = tk.Button(
            btn_row1,
            text="+5s",
            font=("Segoe UI", 9, "bold"),
            activebackground="#cbd5e1",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=8,
            pady=4,
            command=lambda: self.seek_relative(5.0),
        )
        self.fwd5_btn.pack(side=tk.LEFT, padx=2)
        set_btn_enabled(self.fwd5_btn, False, normal_bg="#e2e8f0", normal_fg="#0f172a")

        # Transport row 2: Speed & Bounding Box Toggle
        btn_row2 = tk.Frame(controls_box, bg="#0f172a")
        btn_row2.pack(fill=tk.X)

        bbox_chk = tk.Checkbutton(
            btn_row2,
            text="Highlight Event Bounding Box",
            variable=self._show_bbox_var,
            bg="#0f172a",
            fg="#f8fafc",
            selectcolor="#1e293b",
            activebackground="#0f172a",
            activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            command=self._render_cached_frame,
        )
        bbox_chk.pack(side=tk.LEFT)

        speed_frame = tk.Frame(btn_row2, bg="#0f172a")
        speed_frame.pack(side=tk.RIGHT)
        tk.Label(
            speed_frame,
            text="Playback Speed:",
            bg="#0f172a",
            fg="#f8fafc",
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.speed_var = tk.StringVar(value="1.0x")
        speed_combo = ttk.Combobox(
            speed_frame,
            textvariable=self.speed_var,
            values=["0.5x", "1.0x", "1.5x", "2.0x"],
            width=6,
            state="readonly",
        )
        speed_combo.pack(side=tk.LEFT)
        speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        # 4. Large, Expansible Video Viewport Canvas (fills all vertical & horizontal space!)
        self.canvas_frame = tk.Frame(
            self,
            bg="#090c10",
            height=460,
            highlightbackground="#475569",
            highlightthickness=1,
        )
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.canvas_frame.pack_propagate(False)

        self.video_label = tk.Label(
            self.canvas_frame,
            bg="#090c10",
            fg="#cbd5e1",
            font=("Segoe UI", 10),
            text=(
                "📺 Video Player Viewport (Resizable)\n\n"
                "Click or use Up/Down arrows in the event list\n"
                "to immediately play the recording here.\n\n"
                "• Drag the splitter bar between the Event List and Video Panel\n"
                "  to allocate space horizontally or vertically.\n"
                "• Playback automatically starts at the nearest I-frame\n"
                "  no later than 2 seconds before the event."
            ),
            justify=tk.CENTER,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Re-scale video dynamically whenever the user resizes the panel or drags the splitter
        self.canvas_frame.bind("<Configure>", self._on_canvas_configure)

    def _on_canvas_configure(self, evt: tk.Event) -> None:
        new_w = max(320, evt.width - 24)
        self.sub_lbl.configure(wraplength=new_w)
        self.info_text.configure(wraplength=new_w)
        if self._last_raw_frame is not None and not self._is_playing:
            if self._resize_after_id is not None:
                try:
                    self.after_cancel(self._resize_after_id)
                except Exception:
                    pass
            self._resize_after_id = self.after(25, self._render_cached_frame)

    # ------------------------------------------------------------------
    # Event Loading & I-Frame Seeking
    # ------------------------------------------------------------------

    def load_event(self, event: DetectedEvent, auto_play: bool = True) -> None:
        """
        Load `event` into the embedded video player, seek to the nearest
        preceding I-frame no later than `event.start_time_sec - 2.0`, and play.
        """
        self.stop_playback_loop()
        self.event = event

        if not os.path.exists(event.file_path):
            self.title_lbl.configure(text=f"Event #{event.event_id}: File Not Found")
            self.video_label.configure(
                image="",
                text=f"Could not find video file:\n{event.file_path}",
            )
            return

        if (
            self._cap is None
            or not self._cap.isOpened()
            or self._current_file_path != event.file_path
        ):
            if self._cap is not None:
                self._cap.release()
            self._cap = cv2.VideoCapture(event.file_path)
            self._current_file_path = event.file_path

        if not self._cap.isOpened():
            messagebox.showerror(
                "Playback Error",
                f"Could not open video stream:\n{event.file_path}",
                parent=self,
            )
            return

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if (fps and fps > 0) else 25.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration_sec = (
            self._total_frames / self._fps if self._total_frames > 0 else 0.0
        )

        self.seek_slider.configure(to=max(0.1, self._duration_sec))
        self.time_total_lbl.configure(text=format_timestamp(self._duration_sec))

        set_btn_enabled(self.play_btn, True, normal_bg="#1d4ed8", normal_fg="#ffffff")
        set_btn_enabled(self.preroll_btn, True, normal_bg="#38bdf8", normal_fg="#090d16")
        set_btn_enabled(self.exact_btn, True, normal_bg="#e2e8f0", normal_fg="#0f172a")
        set_btn_enabled(self.back5_btn, True, normal_bg="#e2e8f0", normal_fg="#0f172a")
        set_btn_enabled(self.fwd5_btn, True, normal_bg="#e2e8f0", normal_fg="#0f172a")

        self.seek_to_event_minus_2s()

        self.title_lbl.configure(
            text=f"Event #{event.event_id}: {event.event_type} — {event.camera_name}"
        )
        self.sub_lbl.configure(
            text=(
                f"File: {event.file_name}  |  Event: {event.start_time_str}–{event.end_time_str}  |  "
                f"I-Frame Start (≤ -2s): {format_timestamp(self._iframe_start_sec)}"
            )
        )
        self.preroll_btn.configure(
            text=f"⏮ I-Frame ({format_timestamp(self._iframe_start_sec)})"
        )

        details = (
            f"Camera: {event.camera_name}   •   File: {event.file_name}   •   "
            f"Window: {event.start_time_str}–{event.end_time_str} ({event.duration_str})\n"
            f"Displacement Score: {event.displacement_score_str} of pixels affected   •   "
            f"Peak Area: {event.max_area_pct:.1f}%   •   Confidence: {event.confidence_str}\n"
            f"I-Frame Start: {format_timestamp(self._iframe_start_sec)} (frame #{self._iframe_start_idx}, "
            f"{event.start_time_sec - self._iframe_start_sec:.2f}s before event)   •   "
            f"{event.description}"
        )
        self.info_text.configure(text=details)

        if auto_play:
            self._is_playing = True
            self.play_btn.configure(text="⏸ Pause", bg="#1d4ed8", fg="#ffffff")
            self._schedule_next_frame()
        else:
            self._is_playing = False
            self.play_btn.configure(text="▶ Play", bg="#15803d", fg="#ffffff")

    def seek_to_event_minus_2s(self) -> None:
        """
        Seek to the nearest preceding I-frame no later than
        `max(0.0, event.start_time_sec - 2.0)`.
        """
        if self.event is None or self._cap is None or not self._cap.isOpened():
            return
        max_target_sec = max(0.0, self.event.start_time_sec - 2.0)
        iframe_idx = seek_to_iframe_at_or_before(
            self._cap, max_target_sec, self._fps, self._total_frames
        )
        self._iframe_start_idx = iframe_idx
        self._iframe_start_sec = iframe_idx / self._fps
        self._current_frame_idx = iframe_idx
        self._read_and_render_single_frame()

    def _seek_to_exact_event_iframe(self) -> None:
        """Seek to the nearest preceding I-frame at or before `event.start_time_sec`."""
        if self.event is None:
            return
        self.seek_to_seconds(self.event.start_time_sec, align_to_iframe=True)

    def seek_relative(self, delta_sec: float) -> None:
        current_sec = self._current_frame_idx / self._fps
        self.seek_to_seconds(current_sec + delta_sec, align_to_iframe=True)

    def seek_to_seconds(self, target_sec: float, align_to_iframe: bool = True) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        clamped_sec = max(0.0, min(self._duration_sec, target_sec))
        if align_to_iframe:
            target_frame = seek_to_iframe_at_or_before(
                self._cap, clamped_sec, self._fps, self._total_frames
            )
        else:
            target_frame = int(round(clamped_sec * self._fps))
            target_frame = max(0, min(max(0, self._total_frames - 1), target_frame))
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        self._current_frame_idx = target_frame
        self._read_and_render_single_frame()

    def toggle_play_pause(self) -> None:
        if self.event is None:
            return
        self._is_playing = not self._is_playing
        if self._is_playing:
            if (
                self._total_frames > 0
                and self._current_frame_idx >= self._total_frames - 2
            ):
                self.seek_to_event_minus_2s()
            self.play_btn.configure(text="⏸ Pause", bg="#1d4ed8", fg="#ffffff")
            self._schedule_next_frame()
        else:
            self.play_btn.configure(text="▶ Play", bg="#15803d", fg="#ffffff")
            self.stop_playback_loop()

    def stop_playback_loop(self) -> None:
        self._is_playing = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def release(self) -> None:
        self.stop_playback_loop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _on_speed_change(self, _evt: object = None) -> None:
        val = self.speed_var.get().replace("x", "")
        try:
            self._speed_multiplier = max(0.25, float(val))
        except ValueError:
            self._speed_multiplier = 1.0

    def _on_slider_press(self, _evt: object) -> None:
        self._user_dragging_slider = True

    def _on_slider_move(self, val: str) -> None:
        if self._user_dragging_slider:
            try:
                sec = float(val)
                self.time_current_lbl.configure(text=format_timestamp(sec))
            except ValueError:
                pass

    def _on_slider_release(self, _evt: object) -> None:
        self._user_dragging_slider = False
        self.seek_to_seconds(self.seek_var.get(), align_to_iframe=True)

    def _schedule_next_frame(self) -> None:
        if not self._is_playing or not self.winfo_exists():
            return
        delay_ms = max(
            10, int(round((1000.0 / self._fps) / self._speed_multiplier))
        )
        self._after_id = self.after(delay_ms, self._playback_step)

    def _playback_step(self) -> None:
        if not self._is_playing or not self.winfo_exists():
            return
        ok = self._read_and_render_single_frame()
        if not ok:
            self._is_playing = False
            self.play_btn.configure(
                text="▶ Replay (I-Frame ≤ -2s)", bg="#15803d", fg="#ffffff"
            )
            return
        self._schedule_next_frame()

    def _read_and_render_single_frame(self) -> bool:
        if self._cap is None or not self._cap.isOpened() or self.event is None:
            return False

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False

        self._last_raw_frame = frame
        self._current_frame_idx = max(
            0, int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        )
        ft = int(self._cap.get(cv2.CAP_PROP_FRAME_TYPE))
        if 32 <= ft <= 126:
            self._last_frame_type_char = chr(ft)

        self._render_cached_frame()
        return True

    def _render_cached_frame(self) -> None:
        if self._last_raw_frame is None or self.event is None or not self.winfo_exists():
            return

        current_sec = max(0.0, self._current_frame_idx / self._fps)

        if self._last_frame_type_char == "I":
            self.iframe_badge_lbl.configure(
                text=f"Frame #{self._current_frame_idx} [I-Frame]",
                bg="#34d399",
                fg="#052e16",
            )
        else:
            self.iframe_badge_lbl.configure(
                text=f"Frame #{self._current_frame_idx} [{self._last_frame_type_char}-Frame]",
                bg="#7dd3fc",
                fg="#090d16",
            )

        vis = self._last_raw_frame.copy()
        h, w = vis.shape[:2]

        in_event_window = (
            self.event.start_time_sec - 0.15
            <= current_sec
            <= self.event.end_time_sec + 0.25
        )
        in_preroll_window = (
            self._iframe_start_sec
            <= current_sec
            < self.event.start_time_sec - 0.15
        )

        if self._show_bbox_var.get() and in_event_window:
            bx, by, bw, bh = self.event.bounding_box
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 215, 255), 2)
            tag = f"EVENT: {self.event.event_type} ({self.event.confidence_str})"
            cv2.rectangle(
                vis,
                (bx, max(0, by - 24)),
                (min(w, bx + max(bw, 210)), by),
                (0, 215, 255),
                -1,
            )
            cv2.putText(
                vis,
                tag,
                (bx + 5, max(16, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (15, 20, 25),
                1,
                cv2.LINE_AA,
            )
        elif in_preroll_window:
            countdown = max(0.0, self.event.start_time_sec - current_sec)
            banner = f"I-FRAME PRE-ROLL: '{self.event.event_type}' in {countdown:.1f}s"
            cv2.rectangle(vis, (10, 10), (410, 38), (15, 118, 110), -1)
            cv2.putText(
                vis,
                banner,
                (16, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        canvas_w = max(320, self.canvas_frame.winfo_width() - 8)
        canvas_h = max(200, self.canvas_frame.winfo_height() - 8)
        scale = min(canvas_w / float(w), canvas_h / float(h))
        disp_w = max(160, int(round(w * scale)))
        disp_h = max(90, int(round(h * scale)))

        resized = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._photo_image = ImageTk.PhotoImage(image=pil_img)
        self.video_label.configure(image=self._photo_image, text="")

        self.time_current_lbl.configure(text=format_timestamp(current_sec))
        if not self._user_dragging_slider:
            self.seek_var.set(current_sec)
