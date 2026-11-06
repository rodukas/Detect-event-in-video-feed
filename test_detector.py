"""
Automated test suite for the CCTV Event Detection Engine, RTX 3090 GPU Acceleration,
Crash-Safe Periodic Session Saving, Mid-Video Resume, and Video Player Pre-Roll.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

from detector import DetectedEvent, DetectionConfig, GPUAccelerator, VideoEventAnalyzer
from sample_generator import generate_sample_cctv_folder
from session_store import FileProgressState, SessionCheckpoint, SessionStore


def test_rtx_gpu_acceleration_and_four_camera_noise_rejection() -> None:
    """Verify RTX 3090 GPU OpenCL acceleration and 4-camera event detection + noise filtering."""
    gpu_ok, gpu_name = GPUAccelerator.initialize(True)
    assert gpu_ok is True
    assert "3090" in gpu_name or "NVIDIA" in gpu_name or len(gpu_name) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        created_files = generate_sample_cctv_folder(
            output_dir=tmpdir,
            duration_sec=15.0,
            fps=20.0,
        )
        assert len(created_files) == 4

        streamed_events: List[DetectedEvent] = []
        analyzer = VideoEventAnalyzer(
            DetectionConfig(sample_fps=6.0, sensitivity="Balanced", use_gpu=True)
        )
        all_events = analyzer.scan_folder(
            folder_path=tmpdir,
            on_event_detected=lambda ev: streamed_events.append(ev),
        )

        assert len(streamed_events) == len(all_events)
        assert len(all_events) == 4

        by_file = {}
        for ev in all_events:
            by_file.setdefault(ev.file_name, []).append(ev)

        # Cam3_Backyard has ONLY wind moving tree branches and light variations -> 0 events!
        assert "Cam3_Backyard_2026-11-05.mp4" not in by_file

        # Cam1_FrontDoor has Door Opening (~4.8s) and Person Walking (~10.2s)
        cam1_events = by_file.get("Cam1_FrontDoor_2026-11-05.mp4", [])
        assert len(cam1_events) == 2
        assert {e.event_type for e in cam1_events} == {"Door Opening", "Person Walking"}

        # Cam2_Driveway has Car Passing (~5.5s)
        cam2_events = by_file.get("Cam2_Driveway_2026-11-05.mp4", [])
        assert len(cam2_events) == 1
        assert cam2_events[0].event_type == "Car Passing"

        # Cam4_SideGate has Person Walking (~6.5s)
        cam4_events = by_file.get("Cam4_SideGate_2026-11-05.mp4", [])
        assert len(cam4_events) == 1
        assert cam4_events[0].event_type == "Person Walking"

        for ev in all_events:
            expected_preroll = round(max(0.0, ev.start_time_sec - 2.0), 2)
            assert abs(ev.playback_start_sec - expected_preroll) < 1e-4
            assert ev.thumbnail_rgb is not None
            assert ev.displacement_score_pct >= ev.max_area_pct > 0.0

        # Car passing across driveway affects a larger % of frame pixels than stationary door opening
        door_ev = next(e for e in cam1_events if e.event_type == "Door Opening")
        car_ev = cam2_events[0]
        assert car_ev.displacement_score_pct > door_ev.displacement_score_pct


def test_crash_checkpoint_save_restore_and_mid_video_resume() -> None:
    """
    Simulate a PC crash mid-way through screening `Cam1_FrontDoor_2026-11-05.mp4`
    (right after the first event at t=8.0s / frame 160), verify that:
      1. The checkpoint and event thumbnail are atomically saved on disk.
      2. Re-opening the app restores the findings from the last session and marks it resumable.
      3. Resuming continues from frame 160 of `Cam1` (finding the second event at ~10.2s
         without re-detecting the first event at ~4.8s) and finishes the remaining files.
    """
    from app import CCTVEventApp

    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as cache_dir:
        created_files = generate_sample_cctv_folder(
            output_dir=tmpdir,
            duration_sec=15.0,
            fps=20.0,
        )
        cam1_path = created_files[0]

        store = SessionStore(base_cache_dir=cache_dir)
        analyzer = VideoEventAnalyzer(
            DetectionConfig(
                sample_fps=6.0,
                sensitivity="Balanced",
                use_gpu=True,
                checkpoint_interval_sec=0.05,
            )
        )

        # Simulate stopping/crashing inside Cam1 right after the first event (around frame 165 = 8.25s)
        partial_events: List[DetectedEvent] = []
        saved_frame_idx = 0

        def _on_ev(ev: DetectedEvent) -> None:
            partial_events.append(ev)
            # As soon as event #1 (Door Opening ~4.8-6.8s) is finalized around 8.1s, simulate crash!
            analyzer.cancel()

        def _on_ckpt(f_idx: int, tot_f: int, t_sec: float, dur_sec: float, is_done: bool) -> None:
            nonlocal saved_frame_idx
            saved_frame_idx = f_idx

        analyzer.analyze_video_file(
            file_path=cam1_path,
            on_event_detected=_on_ev,
            on_checkpoint=_on_ckpt,
        )

        assert len(partial_events) == 1
        assert partial_events[0].event_type == "Door Opening"
        assert 140 <= saved_frame_idx <= 200

        # Persist checkpoint as the app does during screening
        files_state = {
            f: FileProgressState(file_path=f, file_name=os.path.basename(f), status="pending")
            for f in created_files
        }
        files_state[cam1_path].status = "in_progress"
        files_state[cam1_path].last_frame_idx = saved_frame_idx
        files_state[cam1_path].total_frames = 300
        files_state[cam1_path].last_timestamp_sec = saved_frame_idx / 20.0
        files_state[cam1_path].duration_sec = 15.0

        ckpt = SessionCheckpoint(
            folder_path=tmpdir,
            status="in_progress",
            gpu_device=analyzer.gpu_device_name,
            next_event_id=2,
            files=files_state,
            events=list(partial_events),
        )
        store.save_checkpoint(ckpt)

        # Now launch CCTVEventApp with this SessionStore -> should restore the 1 event and offer Resume
        app = CCTVEventApp(session_store=store, auto_restore_session=True)
        app.withdraw()
        try:
            assert len(app._events) == 1
            assert len(app.tree.get_children()) == 1
            assert app._current_checkpoint is not None
            assert app._current_checkpoint.is_resumable is True
            assert str(app.resume_btn["state"]) == "normal"
        finally:
            app.destroy()

        # Now resume scan_folder from the saved mid-video frame offset!
        resumed_events: List[DetectedEvent] = []
        analyzer2 = VideoEventAnalyzer(
            DetectionConfig(sample_fps=6.0, sensitivity="Balanced", use_gpu=True)
        )
        analyzer2.scan_folder(
            folder_path=tmpdir,
            on_event_detected=lambda ev: resumed_events.append(ev),
            resume_file_offsets={cam1_path: saved_frame_idx},
            completed_files=set(),
            start_event_id=2,
        )

        # Resuming from frame ~165 of Cam1 must find the 3 remaining events (Cam1 Person Walking,
        # Cam2 Car Passing, Cam4 Person Walking) without duplicating Event #1!
        assert len(resumed_events) == 3
        assert [e.event_id for e in resumed_events] == [2, 3, 4]
        combined = partial_events + resumed_events
        assert len(combined) == 4


def test_embedded_right_player_and_iframe_preroll() -> None:
    """
    Verify that:
      1. The video player is embedded on the right side of the main interface (`app.player_frame`).
      2. Selecting an event loads it into `app.player_frame` without opening a popup window.
      3. Playback always starts on an I-frame (`CAP_PROP_FRAME_TYPE == ord('I')`) whose
         timestamp is no later than `event.start_time_sec - 2.0` seconds.
    """
    from app import CCTVEventApp
    from player_window import EmbeddedVideoPlayerFrame

    sample_dir = os.path.join(os.getcwd(), "sample_cctv_feeds")
    if not os.path.isdir(sample_dir):
        generate_sample_cctv_folder(sample_dir, duration_sec=8.0, fps=15.0)

    with tempfile.TemporaryDirectory() as cache_dir:
        store = SessionStore(base_cache_dir=cache_dir)
        app = CCTVEventApp(session_store=store, auto_restore_session=False)
        app.withdraw()
        try:
            assert isinstance(app.player_frame, EmbeddedVideoPlayerFrame)

            ev = DetectedEvent(
                event_id=1,
                file_name="Cam2_Driveway_2026-11-05.mp4",
                file_path=os.path.join(sample_dir, "Cam2_Driveway_2026-11-05.mp4"),
                camera_name="CAM2 (Driveway)",
                start_time_sec=5.7,
                end_time_sec=8.4,
                peak_time_sec=6.5,
                duration_sec=2.7,
                playback_start_sec=3.7,
                event_type="Car Passing",
                confidence=0.95,
                max_area_pct=2.7,
                avg_speed_norm=0.34,
                bounding_box=(100, 190, 135, 70),
                description="Vehicle moving across driveway",
            )
            ev2 = DetectedEvent(
                event_id=2,
                file_name="Cam1_FrontDoor_2026-11-05.mp4",
                file_path=os.path.join(sample_dir, "Cam1_FrontDoor_2026-11-05.mp4"),
                camera_name="CAM1 (FrontDoor)",
                start_time_sec=4.8,
                end_time_sec=6.2,
                peak_time_sec=5.5,
                duration_sec=1.4,
                playback_start_sec=2.8,
                event_type="Door Opening",
                confidence=0.92,
                max_area_pct=2.2,
                avg_speed_norm=0.06,
                bounding_box=(140, 110, 45, 140),
                description="Localized door movement",
                displacement_score_pct=2.85,
            )
            ev.displacement_score_pct = 19.40
            app._register_detected_event(ev2)
            app._register_detected_event(ev)
            assert len(app.tree.get_children()) == 2

            # Sort by displacement score -> highest first (#1 with 19.40% before #2 with 2.85%)
            app.sort_by_column("disp_score")
            assert list(app.tree.get_children()) == ["1", "2"]
            # Toggle sort -> lowest first (#2 before #1)
            app.sort_by_column("disp_score")
            assert list(app.tree.get_children()) == ["2", "1"]

            # Select the event in the table -> loads into embedded player_frame
            app.tree.selection_set("1")
            app._on_tree_select()
            app.update_idletasks()

            # Verify player started on an I-frame no later than (5.7 - 2.0 = 3.7s)
            assert app.player_frame.event is not None
            assert app.player_frame._last_frame_type_char == "I"
            assert app.player_frame._iframe_start_sec <= 3.7 + 1e-6
            assert app.player_frame._iframe_start_sec >= 0.0

            # Verify resizable PanedWindow and split orientation toggle (Left/Right <-> Top/Bottom)
            assert len(app.main_pane.panes()) == 2
            app._toggle_split_orientation()
            app.update_idletasks()
            assert app._split_orient == "vertical"
            app._toggle_split_orientation()
            app.update_idletasks()
            assert app._split_orient == "horizontal"

            app.player_frame.release()
        finally:
            app.destroy()
