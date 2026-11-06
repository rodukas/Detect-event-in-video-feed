"""
Home CCTV 4-Camera Event Analyzer & Video Review Desktop Application.

Features:
  - NVIDIA RTX 3090 GPU hardware acceleration (OpenCV OpenCL UMat + async frame prefetching)
  - Crash-safe periodic progress saving (every 4 seconds & on every event/file)
  - Automatic restoration of findings from the last session on startup with one-click
    Resume from the exact file & video timestamp of the last save
  - Embedded right-hand video player frame (no popup windows blocking the event list)
    that automatically starts playback on the nearest preceding I-frame no later than
    2 seconds prior to the event.
"""

from __future__ import annotations

import csv
import os
import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from detector import (
    DetectedEvent,
    DetectionConfig,
    GPUAccelerator,
    VideoEventAnalyzer,
    format_timestamp,
)
from player_window import EmbeddedVideoPlayerFrame, set_btn_enabled
from sample_generator import generate_sample_cctv_folder
from session_store import FileProgressState, SessionCheckpoint, SessionStore


class CCTVEventApp(tk.Tk):
    """Main application window for screening CCTV MP4 folders and inspecting events."""

    def __init__(
        self,
        initial_folder: Optional[str] = None,
        session_store: Optional[SessionStore] = None,
        auto_restore_session: bool = True,
    ) -> None:
        super().__init__()
        self.config_model = DetectionConfig()
        self.analyzer = VideoEventAnalyzer(self.config_model)
        self.session_store = session_store or SessionStore()

        gpu_badge = self.analyzer.gpu_device_name
        self.title(f"Home CCTV 24/7 Event Analyzer — GPU: {gpu_badge}")
        self.geometry("1520x900")
        self.minsize(1120, 680)
        self.configure(bg="#0f172a")

        self._events: List[DetectedEvent] = []
        self._events_by_id: Dict[int, DetectedEvent] = {}
        self._current_checkpoint: Optional[SessionCheckpoint] = None
        self._ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self._scan_thread: Optional[threading.Thread] = None
        self._is_scanning: bool = False
        self._suppress_select_open: bool = False
        self._split_orient: str = tk.HORIZONTAL
        self._sort_col: str = "id"
        self._sort_desc: bool = False

        self._configure_styles()
        self._build_layout()

        self.protocol("WM_DELETE_WINDOW", self._on_app_close)
        self.after(60, self._process_ui_queue)

        if initial_folder and os.path.isdir(initial_folder):
            self.folder_var.set(initial_folder)
            existing = self.session_store.load_last_session(initial_folder)
            if existing and os.path.normcase(
                os.path.abspath(existing.folder_path)
            ) == os.path.normcase(os.path.abspath(initial_folder)):
                self._restore_checkpoint_into_ui(existing)
            else:
                self.after(200, lambda: self.start_screening(resume=False))
        elif auto_restore_session:
            last_ckpt = self.session_store.load_last_session()
            if last_ckpt is not None and last_ckpt.folder_path:
                self._restore_checkpoint_into_ui(last_ckpt)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Treeview",
            background="#1e293b",
            foreground="#f8fafc",
            fieldbackground="#1e293b",
            rowheight=30,
            font=("Segoe UI", 9),
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#0f172a",
            foreground="#7dd3fc",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padding=(6, 6),
        )
        style.map(
            "Treeview",
            background=[("selected", "#1d4ed8")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#1e293b",
            background="#38bdf8",
            thickness=14,
        )
        # High-contrast Combobox styling (crisp dark text on white field)
        style.configure(
            "TCombobox",
            fieldbackground="#ffffff",
            background="#e2e8f0",
            foreground="#0f172a",
            arrowcolor="#0f172a",
            selectbackground="#1d4ed8",
            selectforeground="#ffffff",
            padding=3,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#ffffff"), ("disabled", "#cbd5e1")],
            foreground=[("readonly", "#0f172a"), ("disabled", "#475569")],
            selectbackground=[("readonly", "#1d4ed8")],
            selectforeground=[("readonly", "#ffffff")],
        )

    def _build_layout(self) -> None:
        # 1. Top Banner & Folder Selection Controls
        top_bar = tk.Frame(self, bg="#1e293b", padx=16, pady=12)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        title_row = tk.Frame(top_bar, bg="#1e293b")
        title_row.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            title_row,
            text="🛡️ Home CCTV 4-Camera Event Detector",
            font=("Segoe UI", 15, "bold"),
            fg="#ffffff",
            bg="#1e293b",
        ).pack(side=tk.LEFT)

        gpu_color = "#34d399" if self.analyzer.gpu_active else "#fbbf24"
        gpu_text = (
            f"⚡ GPU Accelerated: {self.analyzer.gpu_device_name}"
            if self.analyzer.gpu_active
            else "CPU Mode"
        )
        self.gpu_badge_lbl = tk.Label(
            title_row,
            text=gpu_text,
            font=("Segoe UI", 9, "bold"),
            fg="#052e16" if self.analyzer.gpu_active else "#090d16",
            bg=gpu_color,
            padx=10,
            pady=3,
        )
        self.gpu_badge_lbl.pack(side=tk.LEFT, padx=(14, 8))

        self.use_gpu_var = tk.BooleanVar(value=self.config_model.use_gpu)
        gpu_chk = tk.Checkbutton(
            title_row,
            text="Use RTX GPU Acceleration",
            variable=self.use_gpu_var,
            bg="#1e293b",
            fg="#f8fafc",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            command=self._on_toggle_gpu,
        )
        gpu_chk.pack(side=tk.LEFT, padx=4)

        self.demo_btn = tk.Button(
            title_row,
            text="✨ Generate & Screen 4-Camera Sample Feeds",
            font=("Segoe UI", 9, "bold"),
            activebackground="#d97706",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=12,
            pady=5,
            command=self._on_generate_demo_feeds,
        )
        self.demo_btn.pack(side=tk.RIGHT)
        set_btn_enabled(self.demo_btn, True, normal_bg="#fbbf24", normal_fg="#090d16")

        ctrl_row = tk.Frame(top_bar, bg="#1e293b")
        ctrl_row.pack(fill=tk.X)

        tk.Label(
            ctrl_row,
            text="CCTV Video Folder:",
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#1e293b",
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.folder_var = tk.StringVar()
        self.folder_entry = tk.Entry(
            ctrl_row,
            textvariable=self.folder_var,
            font=("Segoe UI", 10),
            bg="#ffffff",
            fg="#0f172a",
            insertbackground="#0f172a",
            relief=tk.SOLID,
            bd=1,
        )
        self.folder_entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 8)
        )

        self.browse_btn = tk.Button(
            ctrl_row,
            text="📂 Browse Folder...",
            font=("Segoe UI", 9, "bold"),
            activebackground="#cbd5e1",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=12,
            pady=5,
            command=self._on_browse_folder,
        )
        self.browse_btn.pack(side=tk.LEFT, padx=(0, 10))
        set_btn_enabled(self.browse_btn, True, normal_bg="#e2e8f0", normal_fg="#0f172a")

        tk.Label(
            ctrl_row,
            text="Noise Filter:",
            font=("Segoe UI", 9, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
        ).pack(side=tk.LEFT, padx=(4, 4))

        self.sensitivity_var = tk.StringVar(value="Balanced")
        sens_combo = ttk.Combobox(
            ctrl_row,
            textvariable=self.sensitivity_var,
            values=["High", "Balanced", "Strict"],
            width=9,
            state="readonly",
        )
        sens_combo.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(
            ctrl_row,
            text="Sample FPS:",
            font=("Segoe UI", 9, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
        ).pack(side=tk.LEFT, padx=(2, 4))

        self.fps_var = tk.StringVar(value="6 FPS (Fast)")
        fps_combo = ttk.Combobox(
            ctrl_row,
            textvariable=self.fps_var,
            values=["4 FPS (Ultra Fast)", "6 FPS (Fast)", "10 FPS (Detailed)"],
            width=15,
            state="readonly",
        )
        fps_combo.pack(side=tk.LEFT, padx=(0, 10))

        self.resume_btn = tk.Button(
            ctrl_row,
            text="⏯ Resume Last Save",
            font=("Segoe UI", 10, "bold"),
            activebackground="#166534",
            activeforeground="#ffffff",
            relief=tk.RAISED,
            bd=1,
            padx=14,
            pady=5,
            command=lambda: self.start_screening(resume=True),
        )
        self.resume_btn.pack(side=tk.LEFT, padx=(0, 6))
        set_btn_enabled(self.resume_btn, False, normal_bg="#15803d", normal_fg="#ffffff")

        self.scan_btn = tk.Button(
            ctrl_row,
            text="🔍 Start Fresh Scan",
            font=("Segoe UI", 10, "bold"),
            activebackground="#1e40af",
            activeforeground="#ffffff",
            relief=tk.RAISED,
            bd=1,
            padx=14,
            pady=5,
            command=lambda: self.start_screening(resume=False),
        )
        self.scan_btn.pack(side=tk.LEFT, padx=(0, 6))
        set_btn_enabled(self.scan_btn, True, normal_bg="#1d4ed8", normal_fg="#ffffff")

        self.stop_btn = tk.Button(
            ctrl_row,
            text="⏹ Pause & Save",
            font=("Segoe UI", 9, "bold"),
            activebackground="#991b1b",
            activeforeground="#ffffff",
            relief=tk.RAISED,
            bd=1,
            padx=12,
            pady=5,
            command=self.stop_screening,
        )
        self.stop_btn.pack(side=tk.LEFT)
        set_btn_enabled(self.stop_btn, False, normal_bg="#b91c1c", normal_fg="#ffffff")

        # 2. Crash Recovery / Previous Session Banner
        self.recovery_banner = tk.Frame(
            self,
            bg="#064e3b",
            padx=16,
            pady=9,
            highlightbackground="#34d399",
            highlightthickness=1,
        )
        self.recovery_title_lbl = tk.Label(
            self.recovery_banner,
            text="💾 Previous Session Restored",
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#064e3b",
        )
        self.recovery_title_lbl.pack(side=tk.LEFT)

        self.recovery_detail_lbl = tk.Label(
            self.recovery_banner,
            text="",
            font=("Segoe UI", 9, "bold"),
            fg="#d1fae5",
            bg="#064e3b",
        )
        self.recovery_detail_lbl.pack(side=tk.LEFT, padx=(12, 0))

        self.banner_restart_btn = tk.Button(
            self.recovery_banner,
            text="🔄 Start Over From Beginning",
            font=("Segoe UI", 9, "bold"),
            bg="#e2e8f0",
            fg="#0f172a",
            activebackground="#cbd5e1",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=10,
            pady=3,
            cursor="hand2",
            command=lambda: self.start_screening(resume=False),
        )
        self.banner_restart_btn.pack(side=tk.RIGHT, padx=(6, 0))

        self.banner_resume_btn = tk.Button(
            self.recovery_banner,
            text="⏯ Resume Where You Left Off",
            font=("Segoe UI", 9, "bold"),
            bg="#34d399",
            fg="#052e16",
            activebackground="#10b981",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=12,
            pady=3,
            cursor="hand2",
            command=lambda: self.start_screening(resume=True),
        )
        self.banner_resume_btn.pack(side=tk.RIGHT, padx=6)

        # 3. Progress & Auto-Save Status Bar
        prog_bar_frame = tk.Frame(self, bg="#0f172a", padx=16, pady=8)
        prog_bar_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(
            value="Ready. Select a folder of MP4 CCTV recordings or click 'Generate & Screen 4-Camera Sample Feeds'."
        )
        status_lbl = tk.Label(
            prog_bar_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 9, "bold"),
            fg="#7dd3fc",
            bg="#0f172a",
        )
        status_lbl.pack(side=tk.LEFT)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            prog_bar_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100.0,
            variable=self.progress_var,
            length=250,
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=(12, 0))

        self.autosave_var = tk.StringVar(value="Auto-save: Ready (every 4s)")
        autosave_lbl = tk.Label(
            prog_bar_frame,
            textvariable=self.autosave_var,
            font=("Segoe UI", 8, "bold"),
            fg="#cbd5e1",
            bg="#0f172a",
        )
        autosave_lbl.pack(side=tk.RIGHT, padx=8)

        # Filter & quick actions bar
        filter_frame = tk.Frame(self, bg="#161f30", padx=16, pady=6)
        filter_frame.pack(fill=tk.X)

        tk.Label(
            filter_frame,
            text="Filter by Type:",
            font=("Segoe UI", 9, "bold"),
            fg="#f8fafc",
            bg="#161f30",
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.filter_type_var = tk.StringVar(value="All Events")
        self.filter_type_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_type_var,
            values=[
                "All Events",
                "Car Passing",
                "Person Walking",
                "Door Opening",
                "Movement Detected",
            ],
            width=18,
            state="readonly",
        )
        self.filter_type_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.filter_type_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._refresh_table_view()
        )

        tk.Label(
            filter_frame,
            text="Filter by Camera:",
            font=("Segoe UI", 9, "bold"),
            fg="#f8fafc",
            bg="#161f30",
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.filter_cam_var = tk.StringVar(value="All Cameras")
        self.filter_cam_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_cam_var,
            values=["All Cameras"],
            width=20,
            state="readonly",
        )
        self.filter_cam_combo.pack(side=tk.LEFT, padx=(0, 16))
        self.filter_cam_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._refresh_table_view()
        )

        tk.Label(
            filter_frame,
            text="💡 Click any column header (e.g. 'Displacement %') to sort",
            font=("Segoe UI", 9),
            fg="#e2e8f0",
            bg="#161f30",
        ).pack(side=tk.LEFT, padx=6)

        export_btn = tk.Button(
            filter_frame,
            text="📥 Export Events CSV",
            font=("Segoe UI", 9, "bold"),
            bg="#e2e8f0",
            fg="#0f172a",
            activebackground="#cbd5e1",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._export_csv,
        )
        export_btn.pack(side=tk.RIGHT, padx=(6, 0))

        self.split_mode_btn = tk.Button(
            filter_frame,
            text="⇅ Split Mode: Left/Right",
            font=("Segoe UI", 9, "bold"),
            bg="#e2e8f0",
            fg="#0f172a",
            activebackground="#cbd5e1",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._toggle_split_orientation,
        )
        self.split_mode_btn.pack(side=tk.RIGHT, padx=4)

        self.sort_disp_btn = tk.Button(
            filter_frame,
            text="🔽 Sort: Highest Displacement %",
            font=("Segoe UI", 9, "bold"),
            bg="#38bdf8",
            fg="#090d16",
            activebackground="#0284c7",
            activeforeground="#000000",
            relief=tk.RAISED,
            bd=1,
            padx=10,
            pady=3,
            cursor="hand2",
            command=lambda: self.sort_by_column("disp_score", force_desc=True),
        )
        self.sort_disp_btn.pack(side=tk.RIGHT, padx=4)

        # 4. Main Resizable Split (`tk.PanedWindow`): Event Table + Video Player Frame
        self.main_pane = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            bg="#475569",
            sashwidth=8,
            sashpad=2,
            sashrelief=tk.RAISED,
            opaqueresize=True,
            bd=0,
        )
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        # Pane 1: Event Table + Summary Bar
        self.left_column = tk.Frame(self.main_pane, bg="#1e293b", width=620, height=300)

        table_container = tk.Frame(self.left_column, bg="#1e293b")
        table_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        columns = (
            "id",
            "camera",
            "file_name",
            "event_type",
            "disp_score",
            "event_time",
            "play_start",
            "duration",
            "area",
            "confidence",
            "details",
        )
        self.tree = ttk.Treeview(
            table_container,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        self._column_headings = {
            "id": ("#", 38, tk.CENTER),
            "camera": ("Camera", 105, tk.W),
            "file_name": ("File Name", 145, tk.W),
            "event_type": ("Event Type", 110, tk.W),
            "disp_score": ("Displacement %", 108, tk.CENTER),
            "event_time": ("Event Time", 82, tk.CENTER),
            "play_start": ("Max Start (-2s)", 90, tk.CENTER),
            "duration": ("Duration", 60, tk.CENTER),
            "area": ("Peak Area", 66, tk.CENTER),
            "confidence": ("Conf.", 52, tk.CENTER),
            "details": ("Characteristics & Motion Summary", 230, tk.W),
        }
        for col, (label, width, anchor) in self._column_headings.items():
            self.tree.heading(
                col,
                text=label,
                command=lambda c=col: self.sort_by_column(c),
            )
            self.tree.column(col, width=width, minwidth=36, anchor=anchor)
        self._update_heading_arrows()

        y_scroll = ttk.Scrollbar(
            table_container, orient=tk.VERTICAL, command=self.tree.yview
        )
        x_scroll = ttk.Scrollbar(
            table_container, orient=tk.HORIZONTAL, command=self.tree.xview
        )
        self.tree.configure(
            yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set
        )

        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self.stats_lbl = tk.Label(
            self.left_column,
            text="Total Events: 0  |  🚗 Cars: 0  |  🚶 People: 0  |  🚪 Doors: 0",
            font=("Segoe UI", 9, "bold"),
            fg="#7dd3fc",
            bg="#0f172a",
            padx=12,
            pady=7,
            anchor="w",
        )
        self.stats_lbl.pack(fill=tk.X, side=tk.BOTTOM)

        # Pane 2: Large, Resizable Embedded Video Player Frame (given majority of space by default)
        self.player_frame = EmbeddedVideoPlayerFrame(self.main_pane, width=880)

        self.main_pane.add(
            self.left_column,
            minsize=320,
            width=580,
            stretch="always",
        )
        self.main_pane.add(
            self.player_frame,
            minsize=460,
            width=880,
            stretch="always",
        )

    def _toggle_split_orientation(self) -> None:
        """
        Toggle the resizable splitter between Horizontal (Left List / Right Video)
        and Vertical (Top List / Bottom Video) so the user can freely allocate
        either horizontal or vertical space between the list and video player.
        """
        self.main_pane.forget(self.left_column)
        self.main_pane.forget(self.player_frame)

        if self._split_orient == tk.HORIZONTAL:
            self._split_orient = tk.VERTICAL
            self.main_pane.configure(orient=tk.VERTICAL)
            self.split_mode_btn.configure(text="⇅ Split Mode: Top/Bottom")
            total_h = max(600, self.main_pane.winfo_height())
            self.main_pane.add(
                self.left_column,
                minsize=160,
                height=int(total_h * 0.36),
                stretch="always",
            )
            self.main_pane.add(
                self.player_frame,
                minsize=280,
                height=int(total_h * 0.64),
                stretch="always",
            )
        else:
            self._split_orient = tk.HORIZONTAL
            self.main_pane.configure(orient=tk.HORIZONTAL)
            self.split_mode_btn.configure(text="⇅ Split Mode: Left/Right")
            total_w = max(1000, self.main_pane.winfo_width())
            self.main_pane.add(
                self.left_column,
                minsize=320,
                width=int(total_w * 0.40),
                stretch="always",
            )
            self.main_pane.add(
                self.player_frame,
                minsize=460,
                width=int(total_w * 0.60),
                stretch="always",
            )

    # ------------------------------------------------------------------
    # GPU Toggle & Session Restoration
    # ------------------------------------------------------------------

    def _on_toggle_gpu(self) -> None:
        self.config_model.use_gpu = bool(self.use_gpu_var.get())
        active, dev_name = GPUAccelerator.initialize(self.config_model.use_gpu)
        self.analyzer.gpu_active = active
        self.analyzer.gpu_device_name = dev_name
        if active:
            self.gpu_badge_lbl.configure(
                text=f"⚡ GPU Accelerated: {dev_name}",
                bg="#34d399",
                fg="#052e16",
            )
        else:
            self.gpu_badge_lbl.configure(
                text="CPU Mode (GPU Disabled)",
                bg="#fbbf24",
                fg="#090d16",
            )

    def _restore_checkpoint_into_ui(self, ckpt: SessionCheckpoint) -> None:
        """
        Populate the table and UI with findings from the saved checkpoint and
        present the option to resume where the last save occurred.
        """
        self._current_checkpoint = ckpt
        self.folder_var.set(ckpt.folder_path)
        self.sensitivity_var.set(ckpt.sensitivity)
        if ckpt.sample_fps <= 4.5:
            self.fps_var.set("4 FPS (Ultra Fast)")
        elif ckpt.sample_fps >= 9.0:
            self.fps_var.set("10 FPS (Detailed)")
        else:
            self.fps_var.set("6 FPS (Fast)")

        self._events.clear()
        self._events_by_id.clear()
        self._suppress_select_open = True
        try:
            for item in self.tree.get_children():
                self.tree.delete(item)
            for ev in ckpt.events:
                self._events.append(ev)
                self._events_by_id[ev.event_id] = ev
                self._insert_event_row(ev)
        finally:
            self._suppress_select_open = False

        cams = sorted({e.camera_name for e in self._events})
        self.filter_cam_combo.configure(values=["All Cameras"] + cams)
        self._update_stats_summary()

        pct = ckpt.overall_progress_pct
        self.progress_var.set(pct)
        self.autosave_var.set(f"Last saved: {ckpt.last_saved_at}")

        self.recovery_banner.pack(
            fill=tk.X, before=self.main_pane, padx=16, pady=(4, 4)
        )
        summary = ckpt.resume_summary_text()

        if ckpt.is_resumable:
            self.recovery_banner.configure(
                bg="#064e3b", highlightbackground="#34d399"
            )
            self.recovery_title_lbl.configure(
                text="💾 Unfinished Session Restored — Ready to Resume!",
                bg="#064e3b",
                fg="#ffffff",
            )
            self.recovery_detail_lbl.configure(
                text=summary, bg="#064e3b", fg="#d1fae5"
            )
            self.banner_resume_btn.pack(side=tk.RIGHT, padx=6)
            set_btn_enabled(
                self.resume_btn, True, normal_bg="#15803d", normal_fg="#ffffff"
            )
            self.status_var.set(
                f"Restored {len(ckpt.events)} event(s) from last session ({pct:.1f}% screened). "
                f"Click 'Resume Where You Left Off' to continue from the last save."
            )
        else:
            self.recovery_banner.configure(
                bg="#1e293b", highlightbackground="#38bdf8"
            )
            self.recovery_title_lbl.configure(
                text="✅ Previous Completed Session Loaded",
                bg="#1e293b",
                fg="#7dd3fc",
            )
            self.recovery_detail_lbl.configure(
                text=summary, bg="#1e293b", fg="#f1f5f9"
            )
            self.banner_resume_btn.pack_forget()
            set_btn_enabled(
                self.resume_btn, False, normal_bg="#15803d", normal_fg="#ffffff"
            )
            self.status_var.set(
                f"Loaded {len(ckpt.events)} event(s) from last completed session ({ckpt.last_saved_at})."
            )

        if self._events:
            first_ev = self._events[0]
            self._suppress_select_open = True
            try:
                self.tree.selection_set(str(first_ev.event_id))
                self.player_frame.load_event(first_ev, auto_play=False)
            finally:
                self._suppress_select_open = False

    # ------------------------------------------------------------------
    # Actions & Screening Workflow
    # ------------------------------------------------------------------

    def _on_browse_folder(self) -> None:
        chosen = filedialog.askdirectory(
            parent=self,
            title="Select Folder Containing CCTV MP4 Video Feeds",
            initialdir=self.folder_var.get() or os.getcwd(),
        )
        if not chosen:
            return

        self.folder_var.set(chosen)
        existing = self.session_store.load_last_session(chosen)
        if (
            existing is not None
            and os.path.normcase(os.path.abspath(existing.folder_path))
            == os.path.normcase(os.path.abspath(chosen))
            and existing.is_resumable
        ):
            self._restore_checkpoint_into_ui(existing)
            ans = messagebox.askyesno(
                "Resume Previous Screening Session?",
                f"A saved session was found for this folder:\n\n"
                f"{existing.resume_summary_text()}\n\n"
                f"Would you like to RESUME where you left off at the time of the last save?\n"
                f"(Click 'No' to start again from the beginning.)",
                parent=self,
            )
            self.start_screening(resume=ans)
        else:
            self.start_screening(resume=False)

    def _on_generate_demo_feeds(self) -> None:
        if self._is_scanning:
            messagebox.showinfo(
                "Busy",
                "Please wait for the current scan to finish or pause it first.",
            )
            return

        demo_dir = os.path.join(os.getcwd(), "sample_cctv_feeds")
        self.status_var.set(
            "Generating 4-camera synthetic CCTV MP4 feeds (with wind, light shifts, car, people, door)..."
        )
        set_btn_enabled(self.demo_btn, False, normal_bg="#fbbf24", normal_fg="#090d16")
        self.update_idletasks()

        def _gen_worker() -> None:
            try:
                generate_sample_cctv_folder(
                    output_dir=demo_dir,
                    duration_sec=15.0,
                    fps=20.0,
                    on_progress=lambda idx, tot, name: self._ui_queue.put(
                        ("gen_progress", idx, tot, name)
                    ),
                )
                self._ui_queue.put(("gen_done", demo_dir))
            except Exception as exc:
                self._ui_queue.put(
                    ("error", f"Failed to generate sample videos: {exc}")
                )

        threading.Thread(target=_gen_worker, daemon=True).start()

    def start_screening(self, resume: bool = False) -> None:
        if self._is_scanning:
            return

        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning(
                "Select a Folder",
                "Please select a valid folder containing MP4 CCTV video files first.",
                parent=self,
            )
            return

        mp4_files = VideoEventAnalyzer.list_mp4_files(folder)
        if not mp4_files:
            messagebox.showinfo(
                "No MP4 Files Found",
                f"No .mp4 files were found in:\n{folder}",
                parent=self,
            )
            return

        self.config_model.sensitivity = self.sensitivity_var.get()
        self.config_model.use_gpu = bool(self.use_gpu_var.get())
        fps_choice = self.fps_var.get()
        if "4" in fps_choice:
            self.config_model.sample_fps = 4.0
        elif "10" in fps_choice:
            self.config_model.sample_fps = 10.0
        else:
            self.config_model.sample_fps = 6.0

        self.recovery_banner.pack_forget()

        completed_files: set[str] = set()
        resume_offsets: Dict[str, int] = {}
        start_event_id = 1

        if (
            resume
            and self._current_checkpoint is not None
            and os.path.normcase(
                os.path.abspath(self._current_checkpoint.folder_path)
            )
            == os.path.normcase(os.path.abspath(folder))
        ):
            ckpt = self._current_checkpoint
            ckpt.status = "in_progress"
            ckpt.gpu_device = self.analyzer.gpu_device_name
            for fpath in mp4_files:
                if fpath not in ckpt.files:
                    ckpt.files[fpath] = FileProgressState(
                        file_path=fpath, file_name=os.path.basename(fpath)
                    )
                else:
                    st = ckpt.files[fpath]
                    if st.status == "completed":
                        completed_files.add(fpath)
                    elif st.last_frame_idx > 0:
                        resume_offsets[fpath] = st.last_frame_idx
            start_event_id = max(
                ckpt.next_event_id,
                (max((e.event_id for e in self._events), default=0) + 1),
            )
        else:
            self.session_store.clear_thumbnails()
            self._events.clear()
            self._events_by_id.clear()
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.filter_cam_var.set("All Cameras")
            self.filter_cam_combo.configure(values=["All Cameras"])
            self._update_stats_summary()

            files_map = {
                fpath: FileProgressState(
                    file_path=fpath,
                    file_name=os.path.basename(fpath),
                    status="pending",
                )
                for fpath in mp4_files
            }
            self._current_checkpoint = SessionCheckpoint(
                folder_path=folder,
                status="in_progress",
                started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                gpu_device=self.analyzer.gpu_device_name,
                sensitivity=self.config_model.sensitivity,
                sample_fps=self.config_model.sample_fps,
                use_gpu=self.config_model.use_gpu,
                next_event_id=1,
                files=files_map,
                events=[],
            )
            self.session_store.save_checkpoint(self._current_checkpoint)

        self._is_scanning = True
        set_btn_enabled(self.scan_btn, False, normal_bg="#1d4ed8", normal_fg="#ffffff")
        set_btn_enabled(self.resume_btn, False, normal_bg="#15803d", normal_fg="#ffffff")
        set_btn_enabled(self.demo_btn, False, normal_bg="#fbbf24", normal_fg="#090d16")
        set_btn_enabled(self.stop_btn, True, normal_bg="#b91c1c", normal_fg="#ffffff")

        action_word = "Resuming" if resume else "Screening"
        self.status_var.set(
            f"{action_word} {len(mp4_files)} MP4 file(s) on {self.analyzer.gpu_device_name}..."
        )

        def _worker() -> None:
            try:
                self.analyzer.scan_folder(
                    folder_path=folder,
                    on_event_detected=lambda ev: self._ui_queue.put(("event", ev)),
                    on_progress=lambda f_idx, f_tot, fname, f_prog, ov_prog, eff_fps: self._ui_queue.put(
                        ("progress", f_idx, f_tot, fname, f_prog, ov_prog, eff_fps)
                    ),
                    resume_file_offsets=resume_offsets,
                    completed_files=completed_files,
                    start_event_id=start_event_id,
                    on_checkpoint=lambda fpath, f_idx, tot_f, t_sec, dur_sec, is_done: self._ui_queue.put(
                        ("checkpoint", fpath, f_idx, tot_f, t_sec, dur_sec, is_done)
                    ),
                )
                was_cancelled = self.analyzer._cancel_requested
                self._ui_queue.put(("scan_done", len(mp4_files), was_cancelled))
            except Exception as exc:
                self._ui_queue.put(("error", str(exc)))

        self._scan_thread = threading.Thread(target=_worker, daemon=True)
        self._scan_thread.start()

    def stop_screening(self) -> None:
        if self._is_scanning:
            self.analyzer.cancel()
            self.status_var.set("Pausing and saving checkpoint...")

    def _process_ui_queue(self) -> None:
        try:
            while True:
                msg = self._ui_queue.get_nowait()
                kind = msg[0]

                if kind == "gen_progress":
                    _, idx, tot, fname = msg
                    self.progress_var.set((idx - 0.5) / max(1, tot) * 100.0)
                    self.status_var.set(
                        f"Generating sample camera feed ({idx}/{tot}): {fname}..."
                    )

                elif kind == "gen_done":
                    _, demo_dir = msg
                    set_btn_enabled(
                        self.demo_btn, True, normal_bg="#fbbf24", normal_fg="#090d16"
                    )
                    self.folder_var.set(demo_dir)
                    self.start_screening(resume=False)

                elif kind == "event":
                    _, event = msg
                    self._register_detected_event(event, persist=True)

                elif kind == "checkpoint":
                    _, fpath, f_idx, tot_f, t_sec, dur_sec, is_done = msg
                    self._handle_checkpoint_update(
                        fpath, f_idx, tot_f, t_sec, dur_sec, is_done
                    )

                elif kind == "progress":
                    _, f_idx, f_tot, fname, f_prog, ov_prog, eff_fps = msg
                    self.progress_var.set(ov_prog * 100.0)
                    speed_str = (
                        f"  •  ⚡ {eff_fps:.0f} video FPS" if eff_fps > 0 else ""
                    )
                    self.status_var.set(
                        f"Screening [{f_idx}/{f_tot}] {fname} ({int(f_prog * 100)}%){speed_str}  —  "
                        f"{len(self._events)} event(s) detected"
                    )

                elif kind == "scan_done":
                    _, total_files, was_cancelled = msg
                    self._is_scanning = False
                    set_btn_enabled(
                        self.scan_btn, True, normal_bg="#1d4ed8", normal_fg="#ffffff"
                    )
                    set_btn_enabled(
                        self.demo_btn, True, normal_bg="#fbbf24", normal_fg="#090d16"
                    )
                    set_btn_enabled(
                        self.stop_btn, False, normal_bg="#b91c1c", normal_fg="#ffffff"
                    )

                    if self._current_checkpoint is not None:
                        self._current_checkpoint.status = (
                            "paused" if was_cancelled else "completed"
                        )
                        self.session_store.save_checkpoint(self._current_checkpoint)
                        self.autosave_var.set(
                            f"Saved: {self._current_checkpoint.last_saved_at}"
                        )
                        if self._current_checkpoint.is_resumable:
                            set_btn_enabled(
                                self.resume_btn,
                                True,
                                normal_bg="#15803d",
                                normal_fg="#ffffff",
                            )
                            self._restore_checkpoint_into_ui(
                                self._current_checkpoint
                            )
                        else:
                            set_btn_enabled(
                                self.resume_btn,
                                False,
                                normal_bg="#15803d",
                                normal_fg="#ffffff",
                            )
                            self.progress_var.set(100.0)

                    if not was_cancelled:
                        self.status_var.set(
                            f"Completed screening {total_files} MP4 file(s) on {self.analyzer.gpu_device_name}. "
                            f"Detected {len(self._events)} genuine event(s) (progress saved)."
                        )

                elif kind == "error":
                    _, err_msg = msg
                    self._is_scanning = False
                    set_btn_enabled(
                        self.scan_btn, True, normal_bg="#1d4ed8", normal_fg="#ffffff"
                    )
                    set_btn_enabled(
                        self.demo_btn, True, normal_bg="#fbbf24", normal_fg="#090d16"
                    )
                    set_btn_enabled(
                        self.stop_btn, False, normal_bg="#b91c1c", normal_fg="#ffffff"
                    )
                    self.status_var.set(f"Error: {err_msg}")
                    messagebox.showerror("Error", err_msg, parent=self)

        except queue.Empty:
            pass
        finally:
            self.after(60, self._process_ui_queue)

    def _handle_checkpoint_update(
        self,
        file_path: str,
        last_frame_idx: int,
        total_frames: int,
        last_time_sec: float,
        duration_sec: float,
        is_file_done: bool,
    ) -> None:
        if self._current_checkpoint is None:
            return
        st = self._current_checkpoint.files.get(file_path)
        if st is None:
            st = FileProgressState(
                file_path=file_path,
                file_name=os.path.basename(file_path),
            )
            self._current_checkpoint.files[file_path] = st

        st.last_frame_idx = int(last_frame_idx)
        st.total_frames = int(total_frames)
        st.last_timestamp_sec = round(float(last_time_sec), 2)
        st.duration_sec = round(float(duration_sec), 2)
        st.status = "completed" if is_file_done else "in_progress"

        self._current_checkpoint.next_event_id = self.analyzer._next_event_id
        self.session_store.save_checkpoint(self._current_checkpoint)
        self.autosave_var.set(
            f"Auto-saved: {self._current_checkpoint.last_saved_at} (@ {format_timestamp(last_time_sec)})"
        )

    def _register_detected_event(
        self, event: DetectedEvent, persist: bool = False
    ) -> None:
        if persist and event.thumbnail_rgb is not None and not event.thumbnail_path:
            event.thumbnail_path = self.session_store.save_thumbnail(
                event.event_id, event.thumbnail_rgb
            )

        self._events.append(event)
        self._events_by_id[event.event_id] = event

        if self._current_checkpoint is not None and persist:
            self._current_checkpoint.events.append(event)
            self._current_checkpoint.next_event_id = max(
                self._current_checkpoint.next_event_id, event.event_id + 1
            )
            self.session_store.save_checkpoint(self._current_checkpoint)
            self.autosave_var.set(
                f"Auto-saved: {self._current_checkpoint.last_saved_at}"
            )

        cams = sorted({e.camera_name for e in self._events})
        self.filter_cam_combo.configure(values=["All Cameras"] + cams)

        if self._event_matches_filter(event):
            if self._sort_col == "id" and not self._sort_desc:
                self._insert_event_row(event)
            else:
                self._refresh_table_view()

        self._update_stats_summary()

    def _event_matches_filter(self, event: DetectedEvent) -> bool:
        t_filter = self.filter_type_var.get()
        if t_filter != "All Events" and event.event_type != t_filter:
            return False
        c_filter = self.filter_cam_var.get()
        if c_filter != "All Cameras" and event.camera_name != c_filter:
            return False
        return True

    def _insert_event_row(self, event: DetectedEvent) -> None:
        self.tree.insert(
            "",
            tk.END,
            iid=str(event.event_id),
            values=(
                f"#{event.event_id}",
                event.camera_name,
                event.file_name,
                event.event_type,
                event.displacement_score_str,
                event.start_time_str,
                event.playback_start_str,
                event.duration_str,
                f"{event.max_area_pct:.1f}%",
                event.confidence_str,
                event.description,
            ),
        )

    def sort_by_column(self, col: str, force_desc: Optional[bool] = None) -> None:
        """
        Sort the event table by column `col`. Numeric score columns (`disp_score`,
        `area`, `confidence`, `duration`) default to Descending (highest first)
        on first click so the strongest events appear at the top.
        """
        high_first_cols = {"disp_score", "area", "confidence", "duration"}
        if force_desc is not None:
            self._sort_col = col
            self._sort_desc = force_desc
        elif self._sort_col == col:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = col
            self._sort_desc = col in high_first_cols

        self._update_heading_arrows()
        self._refresh_table_view()

    def _update_heading_arrows(self) -> None:
        for col, (base_label, _, _) in self._column_headings.items():
            if col == self._sort_col:
                arrow = " ▾" if self._sort_desc else " ▴"
                self.tree.heading(col, text=f"{base_label}{arrow}")
            else:
                self.tree.heading(col, text=base_label)

    def _event_sort_key(self, ev: DetectedEvent) -> object:
        col = self._sort_col
        if col == "id":
            return ev.event_id
        if col == "camera":
            return (ev.camera_name.lower(), ev.start_time_sec)
        if col == "file_name":
            return (ev.file_name.lower(), ev.start_time_sec)
        if col == "event_type":
            return (ev.event_type.lower(), -ev.displacement_score_pct)
        if col == "disp_score":
            return (ev.displacement_score_pct, ev.max_area_pct, ev.confidence)
        if col == "event_time":
            return ev.start_time_sec
        if col == "play_start":
            return ev.playback_start_sec
        if col == "duration":
            return ev.duration_sec
        if col == "area":
            return ev.max_area_pct
        if col == "confidence":
            return (ev.confidence, ev.displacement_score_pct)
        return ev.description.lower()

    def _refresh_table_view(self) -> None:
        prev_sel = self.tree.selection()
        self._suppress_select_open = True
        try:
            for item in self.tree.get_children():
                self.tree.delete(item)
            filtered = [ev for ev in self._events if self._event_matches_filter(ev)]
            filtered.sort(key=self._event_sort_key, reverse=self._sort_desc)
            for ev in filtered:
                self._insert_event_row(ev)
            if prev_sel and self.tree.exists(prev_sel[0]):
                self.tree.selection_set(prev_sel[0])
        finally:
            self._suppress_select_open = False
        self._update_stats_summary()

    def _update_stats_summary(self) -> None:
        total = len(self._events)
        cars = sum(1 for e in self._events if e.event_type == "Car Passing")
        people = sum(1 for e in self._events if e.event_type == "Person Walking")
        doors = sum(1 for e in self._events if e.event_type == "Door Opening")
        self.stats_lbl.configure(
            text=f"Total Events: {total}  |  🚗 Cars: {cars}  |  🚶 People: {people}  |  🚪 Doors: {doors}"
        )

    # ------------------------------------------------------------------
    # Table Row Selection -> Play in Right-Hand Embedded Frame
    # ------------------------------------------------------------------

    def _get_selected_event(self) -> Optional[DetectedEvent]:
        sel = self.tree.selection()
        if not sel:
            return None
        try:
            ev_id = int(sel[0])
            return self._events_by_id.get(ev_id)
        except ValueError:
            return None

    def _on_tree_select(self, _evt: object = None) -> None:
        if self._suppress_select_open:
            return
        event = self._get_selected_event()
        if event is None:
            return
        self.player_frame.load_event(event, auto_play=True)

    def _export_csv(self) -> None:
        if not self._events:
            messagebox.showinfo(
                "No Events",
                "There are no detected events to export yet.",
                parent=self,
            )
            return

        out_path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Detected CCTV Events to CSV",
            defaultextension=".csv",
            initialfile="cctv_detected_events.csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not out_path:
            return

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Event ID",
                    "Camera",
                    "File Name",
                    "File Path",
                    "Event Type",
                    "Displacement Score (%)",
                    "Event Start Time (HH:MM:SS.d)",
                    "Event Start (sec)",
                    "Max Pre-Roll Start -2s (HH:MM:SS.d)",
                    "Max Pre-Roll Start (sec)",
                    "Event End Time (HH:MM:SS.d)",
                    "Duration (sec)",
                    "Peak Area (%)",
                    "Confidence (%)",
                    "Characteristics",
                ]
            )
            for ev in self._events:
                writer.writerow(
                    [
                        ev.event_id,
                        ev.camera_name,
                        ev.file_name,
                        ev.file_path,
                        ev.event_type,
                        ev.displacement_score_pct,
                        ev.start_time_str,
                        ev.start_time_sec,
                        ev.playback_start_str,
                        ev.playback_start_sec,
                        ev.end_time_str,
                        ev.duration_sec,
                        ev.max_area_pct,
                        int(round(ev.confidence * 100)),
                        ev.description,
                    ]
                )

        messagebox.showinfo(
            "Export Complete",
            f"Successfully exported {len(self._events)} event(s) to:\n{out_path}",
            parent=self,
        )

    def _on_app_close(self) -> None:
        if self._is_scanning:
            self.analyzer.cancel()
            if self._current_checkpoint is not None:
                self._current_checkpoint.status = "paused"
                self.session_store.save_checkpoint(self._current_checkpoint)
        self.player_frame.release()
        self.destroy()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Home CCTV 4-Camera Event Detector & Video Review App"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Optional path to a folder containing MP4 CCTV feeds to screen on startup.",
    )
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Generate synthetic 4-camera sample feeds in ./sample_cctv_feeds and screen them.",
    )
    args = parser.parse_args()

    initial_folder = args.folder
    if args.generate_demo:
        demo_dir = os.path.join(os.getcwd(), "sample_cctv_feeds")
        generate_sample_cctv_folder(demo_dir)
        initial_folder = demo_dir

    app = CCTVEventApp(initial_folder=initial_folder)
    app.mainloop()


if __name__ == "__main__":
    main()
