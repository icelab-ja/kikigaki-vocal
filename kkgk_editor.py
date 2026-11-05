import sys
import argparse
import copy
import shutil
import tempfile
from typing import Optional

from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QGraphicsView, QGraphicsScene,
    QMessageBox, QToolButton, QFrame,
    QCheckBox, QTableWidget, QAbstractItemView,
    QSlider, QDoubleSpinBox, QDialog,
    QSizePolicy, QComboBox, QTabWidget
)
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QKeySequence, QShortcut
from PySide6.QtCore import QCoreApplication, Qt, QEvent, QPoint, QPointF, QTimer

import numpy as np
import traceback

from app_info import APP_TITLE, APP_VERSION
from audio_features import FeatureCache
from beat_edit_controller import BeatEditController, BeatEditControllerCallbacks
from candidate_controller import CandidateController, CandidateControllerCallbacks
from checked_controller import CheckedController, CheckedControllerCallbacks
from developer_panel import DeveloperPanelController, DeveloperPanelCallbacks
from edit_log import EditOperationLogger
from note_confidence import ConfidenceController
from beats_view import load_beats_file, BeatEvent
from DataRepresentation import TPQN
from dtpr_adapter import DTPRAdapter
from DualTimePianoRoll import dtpr_class_to_dtpr_file, dtpr_file_to_dtpr_class
from fluidsynth_backend import (
    open_default_midi_output,
    stop_fluidsynth_server,
)
from feature_controller import (
    FeatureBackgroundController,
    FeatureControllerCallbacks,
    extract_feature_cache,
)
from i18n import install_translator
from kkgk_package import (
    KkgkPackageSession,
    package_error_message,
    write_kkgk_package,
)
from midi_file_io import export_notes_to_midi, export_tick_notes_to_midi
from playback import (
    DEFAULT_MIDI_PROGRAM,
    VOLUME_SLIDER_DEFAULT,
    MidiClickPlayer,
    build_stereo_mix,
    load_audio_file,
    max_audio_duration_seconds,
    _estimate_audio_playback_gain,
    _metronome_slider_to_gain,
    _volume_slider_to_gain,
)
from playback_controller import PlaybackSessionCallbacks, PlaybackSessionController
from piano_roll_panel import PianoRollPanel, PianoRollPanelCallbacks
from piano_roll_scene import (
    MTPRGridScene,
    PianoRollScene,
)
from piano_roll_widgets import (
    PianoKeyboardWidget,
    PianoRollView,
    PianoRollViewCallbacks,
    VocalWaveformWidget,
)
from settings_dialog import SettingsDialog

TARGET_WAV_SR = 44100


class HoverVolumeSlider(QWidget):
    def __init__(self, label: str, *, width: int = 42, height: int = 140, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._dragging = False
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_slider_if_unhovered)
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(120)
        self._hover_timer.timeout.connect(self.monitor_hover_state)

        self.button = QToolButton(self)
        self.button.setText(self.tr("Volume"))
        self.button.setToolTip(label)
        self.button.setAutoRaise(True)
        self.button.setFixedWidth(width)
        self.button.setMouseTracking(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.button)

        self.popup = QFrame(None, Qt.Popup | Qt.FramelessWindowHint)
        self.popup.setObjectName("volumeSliderPopup")
        self.popup.setStyleSheet(
            "#volumeSliderPopup {"
            "background: #ffffff;"
            "border: 1px solid #b8c2cc;"
            "border-radius: 4px;"
            "}"
        )
        self.popup.setMouseTracking(True)
        popup_layout = QVBoxLayout(self.popup)
        popup_layout.setContentsMargins(8, 7, 8, 7)
        popup_layout.setSpacing(0)

        self.slider = QSlider(Qt.Vertical, self.popup)
        self.slider.setRange(0, 100)
        self.slider.setValue(VOLUME_SLIDER_DEFAULT)
        self.slider.setFixedHeight(height)
        self.slider.setFixedWidth(30)
        self.slider.setToolTip(label)
        self.slider.setMouseTracking(True)
        self.slider.setStyleSheet(
            "QSlider::groove:vertical {"
            "background: #d9e4f2;"
            "width: 8px;"
            "border-radius: 4px;"
            "}"
            "QSlider::sub-page:vertical {"
            "background: #d9e4f2;"
            "border-radius: 4px;"
            "}"
            "QSlider::add-page:vertical {"
            "background: #2f80d1;"
            "border-radius: 4px;"
            "}"
            "QSlider::handle:vertical {"
            "background: #0f5fa8;"
            "border: 1px solid #073f72;"
            "height: 16px;"
            "margin: 0 -6px;"
            "border-radius: 8px;"
            "}"
        )
        popup_layout.addWidget(self.slider, alignment=Qt.AlignHCenter)

        self.button.clicked.connect(self.show_slider)

        for widget in (self, self.button, self.popup, self.slider):
            widget.installEventFilter(self)

    def _cursor_inside(self, widget) -> bool:
        if widget is None or not widget.isVisible():
            return False
        pos = widget.mapFromGlobal(QCursor.pos())
        return widget.rect().contains(pos)

    def show_slider(self) -> None:
        self._hide_timer.stop()
        self.popup.adjustSize()
        size = self.popup.sizeHint()
        x = max(0, (self.width() - size.width()) // 2)
        y = self.height() + 4
        global_pos = self.mapToGlobal(QPoint(x, y))
        screen = QApplication.screenAt(global_pos)
        if screen is not None:
            available = screen.availableGeometry()
            target_x = min(max(global_pos.x(), available.left()), available.right() - size.width())
            target_y = min(max(global_pos.y(), available.top()), available.bottom() - size.height())
            global_pos = QPoint(target_x, target_y)
        self.popup.move(global_pos)
        self.popup.show()
        self._hover_timer.start()

    def hide_slider_if_unhovered(self) -> None:
        if self._dragging:
            return
        if not self._cursor_inside(self) and not self._cursor_inside(self.popup):
            self.popup.hide()
            self._hide_timer.stop()
            self._hover_timer.stop()

    def schedule_hide_slider(self) -> None:
        if not self._dragging:
            self._hide_timer.start(450)

    def monitor_hover_state(self) -> None:
        if not self.popup.isVisible():
            self._hide_timer.stop()
            self._hover_timer.stop()
            return
        if self._dragging:
            self._hide_timer.stop()
            return
        if self._cursor_inside(self) or self._cursor_inside(self.popup):
            self._hide_timer.stop()
        elif not self._hide_timer.isActive():
            self.schedule_hide_slider()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Enter:
            self.show_slider()
        elif event.type() == QEvent.Leave:
            self.schedule_hide_slider()
        elif watched is self.slider and event.type() == QEvent.MouseButtonPress:
            self._dragging = True
            self._hide_timer.stop()
        elif watched is self.slider and event.type() == QEvent.MouseButtonRelease:
            self._dragging = False
            self.schedule_hide_slider()
        return super().eventFilter(watched, event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self._initialize_state()
        self._initialize_controllers()
        self._initialize_midi_output()
        self._build_main_ui()
        self._connect_ui_signals()
        self._initialize_shortcuts()

    def _initialize_state(self) -> None:
        self.dtpr_adapter: Optional[DTPRAdapter] = None
        self.mtpr_scene: Optional[MTPRGridScene] = None
        self.mtpr_view: Optional[QGraphicsView] = None
        self.mtpr_keyboard: Optional[PianoKeyboardWidget] = None
        self.scene: Optional[PianoRollScene] = None
        self.view: Optional[PianoRollView] = None
        self.keyboard: Optional[PianoKeyboardWidget] = None
        self.waveform_widget: Optional[VocalWaveformWidget] = None

        self.wav1_path: Optional[Path] = None
        self.wav2_path: Optional[Path] = None
        self.wav1_data: Optional[np.ndarray] = None
        self.wav2_data: Optional[np.ndarray] = None
        self.wav1_playback_gain = 1.0
        self.wav2_playback_gain = 1.0
        self.audio_sr: Optional[int] = None

        self.play_wav1_l = True
        self.play_wav1_r = True
        self.play_wav2_l = True
        self.play_wav2_r = True
        self.play_midi_l = True
        self.play_midi_r = True
        self.play_wav1 = True
        self.play_wav2 = True
        self.play_midi = True
        self.play_metronome_click = True
        self.midi_time_mode = "real"
        self.midi_program = DEFAULT_MIDI_PROGRAM

        self.volume_wav1 = _volume_slider_to_gain(VOLUME_SLIDER_DEFAULT)
        self.volume_wav2 = _volume_slider_to_gain(VOLUME_SLIDER_DEFAULT)
        self.volume_midi = _volume_slider_to_gain(VOLUME_SLIDER_DEFAULT)
        self.volume_metronome = _metronome_slider_to_gain(VOLUME_SLIDER_DEFAULT)
        self.playback_speed = 1.0

        self.mtpr_grid_divisions = 12
        self.mtpr_subdivisions_per_beat = 12

        self.edit_logger: Optional[EditOperationLogger] = None
        self.edit_log_path: Optional[Path] = None
        self.current_process_id: Optional[str] = None
        self.current_fun_id: Optional[str] = None
        self.has_unsaved_changes = False
        self.loaded_midi_path: Optional[str] = None
        self.loaded_dtpr_path: Optional[str] = None
        self.kkgk_session = KkgkPackageSession()

        self.conf_ctrl = ConfidenceController(hop_sec=0.01, start_sec=0.0)
        self.enable_confidence_coloring = False
        self.enable_confidence_jump = False
        self.feature_cache: FeatureCache | None = None

        self.beats_path: str | None = None
        self.beat_events: list[BeatEvent] = []
        self.beat_edit_mode = False
        self.BEAT_COLOR = QColor(30, 30, 30)
        self.DOWNBEAT_COLOR = QColor(0, 0, 0)
        self.BEAT_WIDTH = 2
        self.DOWNBEAT_WIDTH = 4

    def _initialize_controllers(self) -> None:
        self.playback_controller = PlaybackSessionController(
            PlaybackSessionCallbacks(
                on_progress=self.on_playback_progress,
                on_finished=self.on_playback_finished,
            )
        )
        self.developer_panel_controller = DeveloperPanelController(
            DeveloperPanelCallbacks(
                get_table=lambda: getattr(self, "dev_table", None),
                get_dtpr_adapter=lambda: self.dtpr_adapter,
                get_ptpr_scene=lambda: self.scene,
                get_mtpr_scene=lambda: self.mtpr_scene,
                get_ptpr_view=lambda: self.view,
                get_mtpr_view=lambda: self.mtpr_view,
                get_ptpr_keyboard=lambda: self.keyboard,
                get_mtpr_keyboard=lambda: self.mtpr_keyboard,
            )
        )
        self.piano_roll_panel = PianoRollPanel(
            PianoRollPanelCallbacks(
                on_confidence_update=self.on_notes_changed_for_confidence,
                on_candidates_update=self.on_notes_changed_refresh_candidates,
                on_ptpr_notes_changed=self.on_ptpr_scene_notes_changed_sync_mtpr,
                on_mtpr_notes_changed=self.on_mtpr_scene_notes_changed_sync_ptpr,
                on_notes_audition_requested=self.on_notes_audition_requested,
                on_playhead_dragged=self.on_playhead_dragged,
                on_keyboard_clicked=self.on_keyboard_key_clicked,
                on_selection_changed=self.on_scene_selection_changed,
                on_dev_update=self.developer_panel_controller.update,
                beat_insert=self.insert_beat_at_time,
                time_sec_to_abs_tick=self._mtpr_time_sec_to_abs_tick_continuous,
                abs_tick_to_time_sec=self._abs_tick_continuous_to_mtpr_time_sec,
            )
        )
        self.feature_background_controller = FeatureBackgroundController(
            FeatureControllerCallbacks(
                get_ptpr_scene=lambda: self.scene,
                get_mtpr_scene=lambda: (
                    self.mtpr_scene if self.dtpr_adapter is not None else None
                ),
                time_sec_to_abs_tick=self._mtpr_time_sec_to_abs_tick_continuous,
            )
        )
        self.candidate_controller = CandidateController(
            CandidateControllerCallbacks(
                get_ptpr_scene=lambda: self.scene,
                get_mtpr_scene=lambda: self.mtpr_scene,
                get_dtpr_adapter=lambda: self.dtpr_adapter,
                get_confidence_data=self._candidate_confidence_data,
                get_pitch_logprobs=lambda: self.conf_ctrl.pitch_rest_logprobs,
                refresh_after_mtpr_apply=self._refresh_after_mtpr_candidate_apply,
            )
        )
        self.beat_edit_controller = BeatEditController(
            BeatEditControllerCallbacks(
                get_scene=lambda: self.scene,
                get_dtpr_adapter=lambda: self.dtpr_adapter,
                get_beat_events=lambda: self.beat_events,
                refresh_after_edit=self._refresh_after_beat_edit,
                sync_beats_from_dtpr=self._sync_beats_from_dtpr,
                show_information=lambda title, message: QMessageBox.information(
                    self,
                    title,
                    message,
                ),
                show_warning=lambda title, message: QMessageBox.warning(
                    self,
                    title,
                    message,
                ),
            )
        )
        self.checked_controller = CheckedController(
            CheckedControllerCallbacks(
                get_dtpr_adapter=lambda: self.dtpr_adapter,
                get_ptpr_scene=lambda: self.scene,
                get_mtpr_scene=lambda: self.mtpr_scene,
                get_edit_logger=lambda: self.edit_logger,
                set_beat_checked=self.beat_edit_controller.set_checked,
                on_changed=self._mark_unsaved_changes,
            )
        )

    def _initialize_midi_output(self) -> None:
        self.midi_out, self.midi_port_name = open_default_midi_output()
        if self.midi_out is None:
            raise RuntimeError("Failed to initialize MIDI output via FluidSynth.")
        self.midi_player = MidiClickPlayer(self.midi_out)
        self.midi_player.set_program(self.midi_program)

    def _build_main_ui(self) -> None:
        self._build_file_actions()
        self._build_menu_bar()

        # ---------- 中央ウィジェット / メインレイアウト ----------
        central = QWidget()
        main_layout = QVBoxLayout(central)
        self.main_layout = main_layout

        self.menu_panel = QWidget()
        self.menu_panel_layout = QVBoxLayout(self.menu_panel)
        self.menu_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.menu_panel_layout.setSpacing(6)
        main_layout.addWidget(self.menu_panel, 0)
        self._build_playback_controls()
        self._build_editor_controls()
        self._build_editor_area()
        self.setCentralWidget(central)

    def _build_file_actions(self) -> None:
        self.load_zip_btn = QAction(self.tr("Load KKGK"), self)
        self.export_zip_btn = QAction(self.tr("Export KKGK"), self)
        self.load_dtpr_btn = QAction(self.tr("Load DTPR"), self)
        self.btn_load_beats = QAction(self.tr("Load Beats"), self)
        self.load_btn = QAction(self.tr("Load MIDI"), self)
        self.load_wav1_btn = QAction(self.tr("Load Audio"), self)
        self.load_wav2_btn = QAction(self.tr("Load Vocal Audio"), self)
        self.load_conf_btn = QAction(self.tr("Load Candidates"), self)
        self.chk_use_lm_quantize = QAction(self.tr("LM Quantization"), self)
        self.chk_use_lm_quantize.setCheckable(True)
        self.chk_use_lm_quantize.setChecked(True)
        self.save_btn = QAction(self.tr("Export Physical-time MIDI"), self)
        self.save_score_midi_btn = QAction(
            self.tr("Export Metrical-time MIDI"), self
        )
        self.export_qpr_btn = QAction(self.tr("Export DTPR"), self)
        self.dev_mode_check = QAction(self.tr("Data Display"), self)
        self.dev_mode_check.setCheckable(True)
        self.status_label = QLabel(self.tr("Load a MIDI file to begin."))
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.mouse_coord_label = QLabel("coord: -")
        self.mouse_coord_label.setFixedWidth(260)
        self.mouse_coord_label.setStyleSheet("font-size: 10px; color: #555;")
        self.mouse_coord_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.save_btn.setEnabled(False)
        self.save_score_midi_btn.setEnabled(False)
        self.export_qpr_btn.setEnabled(False)
        self.export_zip_btn.setEnabled(False)

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu(self.tr("File"))
        open_menu = file_menu.addMenu(self.tr("Open"))
        open_menu.addAction(self.load_zip_btn)

        export_menu = file_menu.addMenu(self.tr("Export"))
        export_menu.addAction(self.export_zip_btn)
        export_menu.addSeparator()
        export_menu.addAction(self.save_btn)
        export_menu.addAction(self.save_score_midi_btn)

        settings_menu = self.menuBar().addMenu(self.tr("Settings"))
        self.settings_action = QAction(self.tr("Settings..."), self)
        self.settings_action.setMenuRole(QAction.NoRole)
        settings_menu.addAction(self.settings_action)
        settings_menu.addSeparator()
        settings_menu.addAction(self.dev_mode_check)

    def _build_playback_controls(self) -> None:
        # ---------- 音源 / 再生バー ----------
        audio_bar = QHBoxLayout()
        self.play_btn = QPushButton(self.tr("Play"))
        self.pause_btn = QPushButton(self.tr("Pause"))
        self.stop_btn = QPushButton(self.tr("Stop"))

        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        audio_bar.addWidget(self.play_btn)
        audio_bar.addWidget(self.pause_btn)
        audio_bar.addWidget(self.stop_btn)

        # ---------- 再生位置バー（進行表示のみ） ----------
        progress_bar_layout = QHBoxLayout()
        self.playback_slider = QSlider(Qt.Horizontal)
        self.playback_slider.setRange(0, 0)  # 後で総時間に合わせて更新
        self.playback_slider.setEnabled(True)
        self.playback_time_label = QLabel("00m00.00s / 00m00.00s")
        self.playback_time_label.setFixedWidth(150)
        self.playback_time_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.playback_time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        progress_bar_layout.addWidget(self.playback_time_label)
        progress_bar_layout.addWidget(self.playback_slider)
        self.menu_panel_layout.addLayout(progress_bar_layout)

        # ---- 出力チャンネル ----
        self.cmb_wav1_route = self._make_route_combo()
        self.cmb_wav2_route = self._make_route_combo()
        self.cmb_midi_route = self._make_route_combo()

        # ---- 音量スライダ（0〜100、ホバー時のみ縦表示）----
        self.volume_wav1_control = HoverVolumeSlider(self.tr("Audio Volume"))
        self.volume_wav2_control = HoverVolumeSlider(self.tr("Vocal Audio Volume"))
        self.volume_midi_control = HoverVolumeSlider(self.tr("MIDI Volume"))
        self.slider_wav1 = self.volume_wav1_control.slider
        self.slider_wav2 = self.volume_wav2_control.slider
        self.slider_midi = self.volume_midi_control.slider

        audio_bar.addWidget(QLabel(self.tr("Audio")))
        audio_bar.addWidget(self.cmb_wav1_route)
        audio_bar.addWidget(self.volume_wav1_control)
        audio_bar.addWidget(QLabel(self.tr("Vocal Audio")))
        audio_bar.addWidget(self.cmb_wav2_route)
        audio_bar.addWidget(self.volume_wav2_control)
        audio_bar.addWidget(QLabel(self.tr("MIDI")))
        audio_bar.addWidget(self.cmb_midi_route)
        audio_bar.addWidget(self.volume_midi_control)

        self.chk_midi_ideal_time = QCheckBox(self.tr("Metrical-time MIDI Playback"))
        self.chk_midi_ideal_time.setLayoutDirection(Qt.RightToLeft)
        self.chk_midi_ideal_time.setChecked(False)
        self.chk_midi_ideal_time.setToolTip(
            self.tr("OFF: Physical-time MIDI Playback / ON: Metrical-time MIDI Playback")
        )
        audio_bar.addWidget(self.chk_midi_ideal_time)

        self.cmb_speed = QComboBox()
        self.cmb_speed.addItem("0.25x", 0.25)
        self.cmb_speed.addItem("0.50x", 0.50)
        self.cmb_speed.addItem("0.75x", 0.75)
        self.cmb_speed.addItem("1.00x", 1.00)
        self.cmb_speed.addItem("1.25x", 1.25)
        self.cmb_speed.addItem("1.50x", 1.50)
        self.cmb_speed.setCurrentIndex(3)
        self.cmb_speed.setFixedWidth(80)
        audio_bar.addWidget(QLabel(self.tr("Speed")))
        audio_bar.addWidget(self.cmb_speed)

        audio_bar.addWidget(self.status_label, 1)
        audio_bar.addStretch(1)
        self.menu_panel_layout.addLayout(audio_bar)

    def _build_editor_controls(self) -> None:
        self.advanced_menu_panel = QWidget()
        self.advanced_menu_layout = QVBoxLayout(self.advanced_menu_panel)
        self.advanced_menu_layout.setContentsMargins(0, 0, 0, 0)
        self.advanced_menu_layout.setSpacing(6)
        self.menu_panel_layout.addWidget(self.advanced_menu_panel)

        # ---------- 共通ツールバー ----------
        common_bar = QHBoxLayout()

        # Undo ボタン
        self.undo_btn = QToolButton()
        self.undo_btn.setText("↺")
        self.undo_btn.setToolTip(self.tr("Undo"))
        self.undo_btn.setFixedSize(32, 32)
        self.undo_btn.setStyleSheet(
            """
            QToolButton {
                font-size: 18px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
            QToolButton:hover {
                background-color: #e6e6e6;
            }
            QToolButton:pressed {
                background-color: #dddddd;
            }
            """
        )

        # Redo ボタン
        self.redo_btn = QToolButton()
        self.redo_btn.setText("↻")
        self.redo_btn.setToolTip(self.tr("Redo"))
        self.redo_btn.setFixedSize(32, 32)
        self.redo_btn.setStyleSheet(
            """
            QToolButton {
                font-size: 18px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
            QToolButton:hover {
                background-color: #e6e6e6;
            }
            QToolButton:pressed {
                background-color: #dddddd;
            }
            """
        )

        common_bar.addWidget(self.undo_btn)
        common_bar.addWidget(self.redo_btn)

        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("＋")
        self.btn_zoom_in.setToolTip(self.tr("Zoom in horizontally"))
        common_bar.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("－")
        self.btn_zoom_out.setToolTip(self.tr("Zoom out horizontally"))
        common_bar.addWidget(self.btn_zoom_out)

        self.btn_vertical_zoom_in = QToolButton()
        self.btn_vertical_zoom_in.setText("↕＋")
        self.btn_vertical_zoom_in.setToolTip(self.tr("Zoom in vertically"))
        common_bar.addWidget(self.btn_vertical_zoom_in)

        self.btn_vertical_zoom_out = QToolButton()
        self.btn_vertical_zoom_out.setText("↕－")
        self.btn_vertical_zoom_out.setToolTip(self.tr("Zoom out vertically"))
        common_bar.addWidget(self.btn_vertical_zoom_out)

        common_bar.addSpacing(8)
        common_bar.addWidget(self.mouse_coord_label)
        self.chk_metronome_click = QCheckBox(self.tr("Beat Sound"))
        self.chk_metronome_click.setChecked(True)
        self.chk_metronome_click.setToolTip(self.tr("Play beat sound at beat times"))
        common_bar.addWidget(self.chk_metronome_click)

        self.volume_metronome_control = HoverVolumeSlider(self.tr("Beat Volume"))
        self.slider_metronome = self.volume_metronome_control.slider
        common_bar.addWidget(self.volume_metronome_control)
        common_bar.addStretch(1)
        self.advanced_menu_layout.addLayout(common_bar)

        # ---------- 領域別タブ ----------
        self.menu_tabs = QTabWidget()
        self.ptpr_menu_tab = QWidget()
        self.mtpr_menu_tab = QWidget()
        self.menu_tabs.addTab(self.mtpr_menu_tab, self.tr("Metrical time"))
        self.menu_tabs.addTab(self.ptpr_menu_tab, self.tr("Physical time"))
        self.menu_tabs.setCurrentWidget(self.mtpr_menu_tab)
        self.advanced_menu_layout.addWidget(self.menu_tabs)

        ptpr_tab_layout = QVBoxLayout(self.ptpr_menu_tab)
        ptpr_tab_layout.setContentsMargins(0, 0, 0, 0)
        ptpr_tab_layout.setSpacing(6)

        mtpr_tab_layout = QVBoxLayout(self.mtpr_menu_tab)
        mtpr_tab_layout.setContentsMargins(0, 0, 0, 0)
        mtpr_tab_layout.setSpacing(6)

        ptpr_row_top = QHBoxLayout()
        mtpr_row_top = QHBoxLayout()

        # 信頼度関係
        self.conf_thr_spin = QDoubleSpinBox()
        self.conf_thr_spin.setRange(0.0, 1.0)
        self.conf_thr_spin.setSingleStep(0.01)
        self.conf_thr_spin.setValue(0.20)

        # self.next_low_conf_btn = QPushButton("Next Low-Confidence")
        self.next_low_conf_btn = QPushButton(self.tr("Next Low Confidence"))

        self._update_confidence_controls_enabled()
        self.chk_show_beats = QCheckBox(self.tr("Beat"))
        self.chk_show_beats.setChecked(True)
        ptpr_row_top.addWidget(self.chk_show_beats)

        self.chk_edit_beats = QCheckBox(self.tr("Beat Edit"))
        self.chk_edit_beats.setChecked(False)
        ptpr_row_top.addWidget(self.chk_edit_beats)

        ptpr_row_top.addWidget(self.conf_thr_spin)
        ptpr_row_top.addWidget(self.next_low_conf_btn)

        self.cmb_candidates = QComboBox()
        # self.cmb_candidates.addItems([
        #     "Candidates: Off",
        #     "Candidates: Onset",
        #     "Candidates: Split",
        # ])
        self.cmb_candidates.addItems([
            self.tr("None"),
            self.tr("Onset"),
            self.tr("Split"),
        ])
        self.cmb_candidates.setCurrentIndex(0)

        ptpr_row_top.addWidget(QLabel(self.tr("Edit Candidate")))
        ptpr_row_top.addWidget(self.cmb_candidates)

        self.chk_pitch_candidates = QCheckBox(self.tr("Pitch"))
        self.chk_pitch_candidates.setChecked(False)
        ptpr_row_top.addWidget(self.chk_pitch_candidates)

        self.chk_show_spec_ptpr = QCheckBox(self.tr("Spectrogram"))
        self.chk_show_spec_ptpr.setChecked(False)
        ptpr_row_top.addWidget(self.chk_show_spec_ptpr)

        self.chk_show_chroma_ptpr = QCheckBox(self.tr("Chroma"))
        self.chk_show_chroma_ptpr.setChecked(False)
        ptpr_row_top.addWidget(self.chk_show_chroma_ptpr)

        self.chk_show_pitch_prob_ptpr = QCheckBox(self.tr("Pitch Probability"))
        self.chk_show_pitch_prob_ptpr.setChecked(False)
        ptpr_row_top.addWidget(self.chk_show_pitch_prob_ptpr)

        ptpr_row_top.addStretch(1)
        ptpr_tab_layout.addLayout(ptpr_row_top)

        self.cmb_mtpr_grid_divisions = QComboBox()
        self.cmb_mtpr_grid_divisions.addItem("4", 4)
        self.cmb_mtpr_grid_divisions.addItem("8", 8)
        self.cmb_mtpr_grid_divisions.addItem("12", 12)
        self.cmb_mtpr_grid_divisions.addItem("24", 24)
        self.cmb_mtpr_grid_divisions.setCurrentIndex(2)
        self.cmb_mtpr_grid_divisions.setFixedWidth(70)
        mtpr_row_top.addWidget(QLabel(self.tr("Divisions")))
        mtpr_row_top.addWidget(self.cmb_mtpr_grid_divisions)

        mtpr_row_top.addWidget(QLabel(self.tr("Candidate")))
        self.cmb_mtpr_candidates = QComboBox()
        self.cmb_mtpr_candidates.addItems([
            self.tr("None"),
            self.tr("Onset"),
            self.tr("Offset"),
            self.tr("Split"),
        ])
        self.cmb_mtpr_candidates.setCurrentIndex(1)
        mtpr_row_top.addWidget(self.cmb_mtpr_candidates)

        self.chk_mtpr_pitch_candidates = QCheckBox(self.tr("Pitch"))
        self.chk_mtpr_pitch_candidates.setChecked(True)
        mtpr_row_top.addWidget(self.chk_mtpr_pitch_candidates)

        self.chk_show_spec_mtpr = QCheckBox(self.tr("Spectrogram"))
        self.chk_show_spec_mtpr.setChecked(False)
        mtpr_row_top.addWidget(self.chk_show_spec_mtpr)

        self.chk_show_chroma_mtpr = QCheckBox(self.tr("Chroma"))
        self.chk_show_chroma_mtpr.setChecked(False)
        mtpr_row_top.addWidget(self.chk_show_chroma_mtpr)

        self.chk_show_pitch_prob_mtpr = QCheckBox(self.tr("Pitch Probability"))
        self.chk_show_pitch_prob_mtpr.setChecked(False)
        mtpr_row_top.addWidget(self.chk_show_pitch_prob_mtpr)

        mtpr_row_top.addStretch(1)
        mtpr_tab_layout.addLayout(mtpr_row_top)

    def _build_editor_area(self) -> None:
        # ---------- 下部: 左 = ピアノロール / 右 = 開発者モードテーブル ----------
        self.editor_layout = QHBoxLayout()
        self.main_layout.addLayout(self.editor_layout, 1)

        # 左側: 初期はプレースホルダ
        self.placeholder = QLabel(self.tr("Piano Roll Display Area"))
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: gray; border: 1px solid #ccc;")
        self.left_widget: QWidget = self.placeholder
        self.editor_layout.addWidget(self.left_widget, 3)

        # 右側: 開発者モード用テーブル
        self.dev_table = QTableWidget()
        self.dev_table.setColumnCount(12)
        self.dev_table.setHorizontalHeaderLabels([
            "Note ID",
            "Pitch",
            "Abs On Tick",
            "On Beat ID",
            "Rel On Tick",
            "Abs Off Tick",
            "Off Beat ID",
            "Rel Off Tick",
            "Ideal On Time",
            "Real On Time",
            "Ideal Off Time",
            "Real Off Time",
        ])
        self.dev_table.horizontalHeader().setStretchLastSection(True)
        self.dev_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dev_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.dev_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.dev_table.setFont(QFont("Consolas", 9))
        self.dev_table.setStyleSheet("background-color: #f9f9f9;")
        self.editor_layout.addWidget(self.dev_table, 1)
        self.dev_table.hide()  # 初期は非表示


    def _connect_ui_signals(self) -> None:
        self.settings_action.triggered.connect(self.on_open_settings_dialog)
        self.load_zip_btn.triggered.connect(self.on_load_zip_package)
        self.export_zip_btn.triggered.connect(self.on_export_zip_package)
        self.load_dtpr_btn.triggered.connect(self.on_load_dtpr)
        self.btn_load_beats.triggered.connect(self.on_load_beats_clicked)
        self.load_btn.triggered.connect(self.on_load_midi)
        self.load_wav1_btn.triggered.connect(self.on_load_wav1)
        self.load_wav2_btn.triggered.connect(self.on_load_wav2)
        self.load_conf_btn.triggered.connect(self.on_load_confidence)
        self.save_btn.triggered.connect(self.on_save_midi)
        self.save_score_midi_btn.triggered.connect(self.on_save_score_midi)
        self.export_qpr_btn.triggered.connect(self.on_export_qpr)

        self.play_btn.clicked.connect(self.on_start_playback)
        self.pause_btn.clicked.connect(self.on_pause_playback)
        self.stop_btn.clicked.connect(self.on_stop_playback)
        self.undo_btn.clicked.connect(self.on_undo_clicked)
        self.redo_btn.clicked.connect(self.on_redo_clicked)
        self.btn_zoom_in.clicked.connect(self.on_zoom_in_clicked)
        self.btn_zoom_out.clicked.connect(self.on_zoom_out_clicked)
        self.btn_vertical_zoom_in.clicked.connect(self.on_vertical_zoom_in_clicked)
        self.btn_vertical_zoom_out.clicked.connect(self.on_vertical_zoom_out_clicked)

        self.dev_mode_check.toggled.connect(self.developer_panel_controller.set_visible)
        self.dev_table.cellClicked.connect(self.developer_panel_controller.on_cell_clicked)
        self.conf_thr_spin.valueChanged.connect(self.on_conf_threshold_changed)
        self.next_low_conf_btn.clicked.connect(self.on_next_low_conf)
        self.chk_show_beats.toggled.connect(self.on_toggle_beats)
        self.chk_edit_beats.toggled.connect(self.on_toggle_beat_edit_mode)
        self.cmb_candidates.currentIndexChanged.connect(self.on_candidate_mode_changed)
        self.chk_pitch_candidates.toggled.connect(self.on_toggle_pitch_candidates)
        self.cmb_mtpr_grid_divisions.currentIndexChanged.connect(
            self.on_mtpr_grid_divisions_changed
        )
        self.cmb_mtpr_candidates.currentIndexChanged.connect(
            self.on_mtpr_candidate_mode_changed
        )
        self.chk_mtpr_pitch_candidates.toggled.connect(
            self.on_toggle_mtpr_pitch_candidates
        )

        self.cmb_wav1_route.currentIndexChanged.connect(self.on_audio_route_changed)
        self.cmb_wav2_route.currentIndexChanged.connect(self.on_audio_route_changed)
        self.cmb_midi_route.currentIndexChanged.connect(self.on_audio_route_changed)
        self.slider_wav1.valueChanged.connect(self.on_slider_wav1_changed)
        self.slider_wav2.valueChanged.connect(self.on_slider_wav2_changed)
        self.slider_midi.valueChanged.connect(self.on_slider_midi_changed)
        self.chk_metronome_click.toggled.connect(self.on_metronome_click_toggled)
        self.slider_metronome.valueChanged.connect(self.on_slider_metronome_changed)
        self.cmb_speed.currentIndexChanged.connect(self.on_speed_combo_changed)
        self.chk_midi_ideal_time.toggled.connect(self.on_midi_time_mode_toggled)
        self.playback_slider.sliderPressed.connect(self.on_slider_pressed)
        self.playback_slider.sliderReleased.connect(self.on_slider_released)
        self.playback_slider.valueChanged.connect(self.on_slider_value_changed)

        self.chk_show_spec_ptpr.toggled.connect(self.on_chk_show_spec_ptpr_toggled)
        self.chk_show_spec_mtpr.toggled.connect(self.on_chk_show_spec_mtpr_toggled)
        self.chk_show_chroma_ptpr.toggled.connect(self.on_chk_show_chroma_ptpr_toggled)
        self.chk_show_chroma_mtpr.toggled.connect(self.on_chk_show_chroma_mtpr_toggled)
        self.chk_show_pitch_prob_ptpr.toggled.connect(
            self.on_chk_show_pitch_prob_ptpr_toggled
        )
        self.chk_show_pitch_prob_mtpr.toggled.connect(
            self.on_chk_show_pitch_prob_mtpr_toggled
        )

        self.on_mtpr_grid_divisions_changed(
            self.cmb_mtpr_grid_divisions.currentIndex()
        )

    def _initialize_shortcuts(self) -> None:
        self.space_play_pause_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_play_pause_shortcut.setContext(Qt.ApplicationShortcut)
        self.space_play_pause_shortcut.setAutoRepeat(False)
        self.space_play_pause_shortcut.activated.connect(self.on_space_play_pause)

        self.save_package_shortcut = QShortcut(QKeySequence.Save, self)
        self.save_package_shortcut.setContext(Qt.ApplicationShortcut)
        self.save_package_shortcut.setAutoRepeat(False)
        self.save_package_shortcut.activated.connect(self.on_export_zip_package)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self.mtpr_candidate_cycle_shortcut = QShortcut(
            QKeySequence(Qt.Key_C),
            self,
        )
        self.mtpr_candidate_cycle_shortcut.setContext(Qt.ApplicationShortcut)
        self.mtpr_candidate_cycle_shortcut.setAutoRepeat(False)
        self.mtpr_candidate_cycle_shortcut.activated.connect(
            self.on_cycle_mtpr_candidate_mode
        )

    @staticmethod
    def _is_undo_key_event(event) -> bool:
        if event.key() != Qt.Key_Z or event.isAutoRepeat():
            return False
        modifiers = event.modifiers()
        command = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier))
        excluded = bool(modifiers & (Qt.ShiftModifier | Qt.AltModifier))
        return command and not excluded

    @staticmethod
    def _is_save_key_event(event) -> bool:
        if event.key() != Qt.Key_S or event.isAutoRepeat():
            return False
        modifiers = event.modifiers()
        command = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier))
        excluded = bool(modifiers & (Qt.ShiftModifier | Qt.AltModifier))
        return command and not excluded

    def eventFilter(self, watched, event):
        if event.type() == QEvent.KeyPress:
            if self._is_undo_key_event(event):
                self.on_undo_clicked()
                event.accept()
                return True
            if self._is_save_key_event(event):
                self.on_export_zip_package()
                event.accept()
                return True
        return super().eventFilter(watched, event)

    @staticmethod
    def _make_route_combo() -> QComboBox:
        combo = QComboBox()
        combo.addItem("LR", "lr")
        combo.addItem("L", "left")
        combo.addItem("R", "right")
        combo.addItem("-", "none")
        combo.setCurrentIndex(0)
        combo.setFixedWidth(58)
        combo.setToolTip(QCoreApplication.translate("MainWindow", "Output Channel"))
        return combo

    def _piano_roll_view_callbacks(self) -> PianoRollViewCallbacks:
        return PianoRollViewCallbacks(
            apply_candidate_by_number=self.apply_candidate_by_number_key,
            toggle_selected_beat_downbeat=self.toggle_selected_beat_downbeat,
            handle_checked_key=self.checked_controller.handle_key,
            delete_selected_beats=self.delete_selected_beats,
            on_x_zoom_changed=self.on_view_x_zoom_changed,
            on_y_zoom_changed=self.on_view_y_zoom_changed,
            update_mouse_coord=self.update_mouse_coord_label,
            clear_mouse_coord=self.clear_mouse_coord_label,
        )

    def _build_beat_events_from_bdb(self, bdb) -> list[BeatEvent]:
        out: list[BeatEvent] = []
        if bdb is None or not hasattr(bdb, "index") or not hasattr(bdb, "beats"):
            return out
        for _, beat_id in list(bdb.index):
            beat_obj = bdb.beats.get(beat_id)
            if beat_obj is None:
                continue
            t = getattr(beat_obj, "beat_time", None)
            if t is None:
                continue
            out.append(
                BeatEvent(
                    time_sec=float(t),
                    is_downbeat=bool(getattr(beat_obj, "is_downbeat", False)),
                    beat_id=int(beat_id),
                    checked=bool(getattr(beat_obj, "checked", False)),
                )
            )
        out.sort(key=lambda e: float(e.time_sec))
        return out

    def _sync_beats_from_dtpr(self) -> None:
        if self.dtpr_adapter is None:
            self.beat_events = []
            self.clear_beats()
            return
        self.beat_events = self._build_beat_events_from_bdb(self.dtpr_adapter.dtpr.BDB)
        self.redraw_beats()

    def _beat_click_events_for_playback(self) -> list[tuple[float, bool]]:
        if self.dtpr_adapter is None:
            return []
        out: list[tuple[float, bool]] = []
        for beat in self.dtpr_adapter.get_beats_sorted():
            try:
                out.append((float(beat.beat_time), bool(beat.is_downbeat)))
            except Exception:
                continue
        out.sort(key=lambda item: float(item[0]))
        return out

    def update_mouse_coord_label(self, scene: PianoRollScene, scene_pos: QPointF) -> None:
        label = getattr(self, "mouse_coord_label", None)
        if label is None:
            return

        x = float(scene_pos.x())
        y = float(scene_pos.y())
        pitch = int(round(float(scene.max_pitch) - y / float(scene.px_per_pitch)))
        pitch = max(int(scene.min_pitch), min(int(scene.max_pitch), pitch))

        if scene._is_ptpr():
            sec = float(scene.x_to_domain_time(x))
            snap = float(scene.snap_domain_time(sec))
            label.setText(f"PTPR x={x:.1f} sec={sec:.3f} snap={snap:.3f} pitch={pitch}")
        else:
            tick = float(scene.x_to_domain_time(x))
            snap_tick = float(scene.snap_domain_time(tick))
            label.setText(f"MTPR x={x:.1f} tick={tick:.1f} snap={snap_tick:.0f} pitch={pitch}")

    def clear_mouse_coord_label(self) -> None:
        label = getattr(self, "mouse_coord_label", None)
        if label is not None:
            label.setText("coord: -")

    def _mark_unsaved_changes(self) -> None:
        if self.dtpr_adapter is not None:
            self.has_unsaved_changes = True

    def _mark_saved_changes(self) -> None:
        self.has_unsaved_changes = False

    def _snapshot_load_state(self) -> dict:
        confidence_state = self.conf_ctrl.snapshot_model_output()
        return {
            "dtpr_adapter": copy.deepcopy(self.dtpr_adapter),
            "beats_path": self.beats_path,
            "beat_events": copy.deepcopy(self.beat_events),
            "loaded_midi_path": self.loaded_midi_path,
            "loaded_dtpr_path": self.loaded_dtpr_path,
            "edit_log_path": self.edit_log_path,
            "current_process_id": self.current_process_id,
            "current_fun_id": self.current_fun_id,
            "has_unsaved_changes": self.has_unsaved_changes,
            **self.kkgk_session.snapshot_metadata(),
            "edit_logger": self.edit_logger,
            "wav1_path": self.wav1_path,
            "wav1_data": self.wav1_data,
            "wav1_playback_gain": self.wav1_playback_gain,
            "wav2_path": self.wav2_path,
            "wav2_data": self.wav2_data,
            "wav2_playback_gain": self.wav2_playback_gain,
            "audio_sr": self.audio_sr,
            "feature_cache": copy.deepcopy(self.feature_cache),
            **confidence_state,
            "status_text": self.status_label.text() if getattr(self, "status_label", None) is not None else "",
            "save_enabled": self.save_btn.isEnabled() if getattr(self, "save_btn", None) is not None else False,
            "export_enabled": self.export_qpr_btn.isEnabled() if getattr(self, "export_qpr_btn", None) is not None else False,
            "export_zip_enabled": self.export_zip_btn.isEnabled() if getattr(self, "export_zip_btn", None) is not None else False,
        }

    def _clear_loaded_package_state(self) -> None:
        self.dtpr_adapter = None
        self.beats_path = None
        self.beat_events = []
        self.loaded_midi_path = None
        self.loaded_dtpr_path = None
        self.kkgk_session.clear_metadata()
        self.edit_logger = None
        self.edit_log_path = None
        self.current_process_id = None
        self.current_fun_id = None
        self.has_unsaved_changes = False
        self.wav1_path = None
        self.wav1_data = None
        self.wav1_playback_gain = 1.0
        self.wav2_path = None
        self.wav2_data = None
        self.wav2_playback_gain = 1.0
        self.audio_sr = None
        self.feature_cache = None
        self.candidate_controller.reset()
        self.conf_ctrl.clear_model_output()
        self.save_btn.setEnabled(False)
        self.save_score_midi_btn.setEnabled(False)
        self.export_qpr_btn.setEnabled(False)
        self.export_zip_btn.setEnabled(False)

    def _reset_editor_to_placeholder(self) -> None:
        self.clear_beats()
        self.piano_roll_panel.clear()
        self._sync_piano_roll_panel_refs()

        if getattr(self, "left_widget", None) is not getattr(self, "placeholder", None):
            try:
                self.editor_layout.removeWidget(self.left_widget)
            except Exception:
                pass
            try:
                self.left_widget.setParent(None)
            except Exception:
                pass
            self.left_widget = self.placeholder
            if self.placeholder.parent() is None:
                self.editor_layout.insertWidget(0, self.left_widget, 3)

    def _restore_load_state(self, state: dict) -> None:
        self.dtpr_adapter = state.get("dtpr_adapter")
        self.beats_path = state.get("beats_path")
        self.beat_events = copy.deepcopy(state.get("beat_events", []))
        self.loaded_midi_path = state.get("loaded_midi_path")
        self.loaded_dtpr_path = state.get("loaded_dtpr_path")
        self.edit_log_path = state.get("edit_log_path")
        self.current_process_id = state.get("current_process_id")
        self.current_fun_id = state.get("current_fun_id")
        self.has_unsaved_changes = bool(state.get("has_unsaved_changes", False))
        self.kkgk_session.restore_metadata(state)
        self.edit_logger = state.get("edit_logger")
        self.wav1_path = state.get("wav1_path")
        self.wav1_data = state.get("wav1_data")
        self.wav1_playback_gain = float(state.get("wav1_playback_gain", 1.0))
        self.wav2_path = state.get("wav2_path")
        self.wav2_data = state.get("wav2_data")
        self.wav2_playback_gain = float(state.get("wav2_playback_gain", 1.0))
        self.audio_sr = state.get("audio_sr")
        self.feature_cache = copy.deepcopy(state.get("feature_cache"))
        self.conf_ctrl.restore_model_output(state)
        self.candidate_controller.reset()

        if self.dtpr_adapter is not None:
            self.setup_pianoroll()
            self._sync_beats_from_dtpr()
        else:
            self._reset_editor_to_placeholder()
            self.save_btn.setEnabled(False)
            self.save_score_midi_btn.setEnabled(False)
            self.export_qpr_btn.setEnabled(False)
            self.export_zip_btn.setEnabled(False)

        self._update_play_button_state()
        self._update_confidence_controls_enabled()
        if getattr(self, "status_label", None) is not None:
            self.status_label.setText(str(state.get("status_text", "")))
        self.save_btn.setEnabled(bool(state.get("save_enabled", False)))
        self.save_score_midi_btn.setEnabled(bool(state.get("save_enabled", False)))
        self.export_qpr_btn.setEnabled(bool(state.get("export_enabled", False)))
        self.export_zip_btn.setEnabled(bool(state.get("export_zip_enabled", False)))

    def _load_dtpr_from_path(
        self,
        file_path: str,
        *,
        update_status: bool = True,
        log_path: Optional[str] = None,
        log_dtpr_path: Optional[str] = None,
        log_inner_path: Optional[str] = None,
        process_id: Optional[str] = None,
        fun_id: Optional[str] = None,
    ) -> None:
        if update_status:
            self.kkgk_session.clear_metadata()
            self.loaded_midi_path = None
            self.current_process_id = None
            self.current_fun_id = None
            self.export_zip_btn.setEnabled(False)

        dtpr = dtpr_file_to_dtpr_class(file_path)
        self.dtpr_adapter = DTPRAdapter(dtpr=dtpr)
        self._sync_dtpr_quantize_delta_from_grid_divisions()
        self.loaded_dtpr_path = file_path
        self.beats_path = None

        log_path = str(log_path) if log_path is not None else str(Path(file_path).with_suffix(".edit_log.jsonl"))
        self.edit_log_path = Path(log_path)
        self.current_process_id = process_id
        self.current_fun_id = fun_id
        session_meta = {
            "dtprPath": log_dtpr_path or file_path,
            "appVersion": APP_VERSION,
            "gridSec": 0.01,
            "minNoteLenSec": 0.02,
        }
        identity_meta = {}
        if process_id:
            identity_meta["processId"] = process_id
        if fun_id:
            identity_meta["funId"] = fun_id
        session_meta = {**identity_meta, **session_meta}
        if log_inner_path:
            session_meta["logPath"] = log_inner_path
        self.edit_logger = EditOperationLogger(log_path, auto_flush=True)
        self.edit_logger.append_session(meta=session_meta)

        self.setup_pianoroll()
        self._sync_beats_from_dtpr()
        if update_status:
            self.status_label.setText(
                self.tr("DTPR loaded: {filename}").format(
                    filename=Path(file_path).name
                )
            )
        self.save_btn.setEnabled(True)
        self.save_score_midi_btn.setEnabled(True)
        self.export_qpr_btn.setEnabled(True)
        self._mark_saved_changes()

    def _load_audio_from_path(self, file_path: str, *, which: int, update_status: bool = True) -> None:
        data, sample_rate = load_audio_file(file_path, TARGET_WAV_SR)
        self.audio_sr = int(sample_rate)

        if which == 1:
            self.wav1_path = Path(file_path)
            self.wav1_data = data
            self.wav1_playback_gain = _estimate_audio_playback_gain(data)
            if update_status:
                self.status_label.setText(
                    self.tr("Audio loaded: {filename}").format(
                        filename=self.wav1_path.name
                    )
                )
        else:
            self.wav2_path = Path(file_path)
            self.wav2_data = data
            self.wav2_playback_gain = _estimate_audio_playback_gain(data)
            if update_status:
                self.status_label.setText(
                    self.tr("Vocal audio loaded: {filename}").format(
                        filename=Path(file_path).name
                    )
                )
            if self.waveform_widget is not None and self.audio_sr is not None:
                self.waveform_widget.set_audio(self.wav2_data, self.audio_sr)
            try:
                self.feature_cache = extract_feature_cache(self.wav2_data, self.audio_sr)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    self.tr("Feature Extraction Error"),
                    self.tr("An error occurred while extracting features:\n{error}").format(
                        error=self.tr(str(e))
                    ),
                )
                self.feature_cache = None
            self.compute_and_set_background_features()

        self._update_scene_time_range()
        self._update_play_button_state()

    def _load_confidence_from_path(self, path: str) -> None:
        self.load_model_output_npy(path)
        self.refresh_candidates_from_selection()
        self._update_confidence_controls_enabled()
        scene = getattr(self, "scene", None)
        if scene is not None:
            self.on_notes_changed_for_confidence()

    def on_load_zip_package(self):
        zip_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select a KKGK File"),
            "",
            "KKGK Files (*.kkgk);;All Files (*.*)",
        )
        if not zip_path:
            return

        snapshot = self._snapshot_load_state()
        temp_dir = None

        try:
            temp_dir, package = self.kkgk_session.prepare_load(zip_path)
            pid = package.pid
            members = package.members
            extracted = package.paths

            self._clear_loaded_package_state()

            dtpr_member = members.get("dtpr")
            log_member = members.get("log")
            zip_log_path = (
                self.kkgk_session.edit_log_path(
                    package_path=zip_path,
                    member=dtpr_member,
                    log_member=log_member,
                    fallback_paths=(self.loaded_midi_path, self.loaded_dtpr_path),
                )
                if dtpr_member
                else None
            )
            if zip_log_path is not None:
                zip_log_path.parent.mkdir(parents=True, exist_ok=True)
                if "log" in extracted:
                    shutil.copyfile(extracted["log"], zip_log_path)
                else:
                    zip_log_path.write_text("", encoding="utf-8")
            process_id = pid.get("processId", "")
            fun_id = pid.get("funId")
            self._load_dtpr_from_path(
                extracted["dtpr"],
                update_status=False,
                log_path=str(zip_log_path) if zip_log_path is not None else None,
                log_dtpr_path=dtpr_member or extracted["dtpr"],
                log_inner_path=log_member,
                process_id=process_id,
                fun_id=fun_id,
            )
            if "audio" in extracted:
                self._load_audio_from_path(extracted["audio"], which=1, update_status=False)
            if "audio_vocals" in extracted:
                self._load_audio_from_path(extracted["audio_vocals"], which=2, update_status=False)
            if "candidate" in extracted:
                self._load_confidence_from_path(extracted["candidate"])
            else:
                self._update_confidence_controls_enabled()

            self.kkgk_session.commit_load(zip_path, temp_dir, members)
            self.export_zip_btn.setEnabled(self.kkgk_session.can_export())
            suffix = f" ({process_id})" if process_id else ""
            self.status_label.setText(
                self.tr("KKGK loaded{suffix}: {filename}").format(
                    suffix=suffix,
                    filename=Path(zip_path).name,
                )
            )
        except Exception as e:
            self.kkgk_session.discard_load(temp_dir)
            self._restore_load_state(snapshot)
            print(f"[KKGK load] failed: {e}", file=sys.stderr)
            traceback.print_exc()
            QMessageBox.critical(
                self,
                self.tr("KKGK Load Error"),
                self.tr("An error occurred while loading the KKGK file:\n{error}").format(
                    error=package_error_message(e, self.tr)
                ),
            )

    def on_load_midi(self):
        if self.dtpr_adapter is None or len(getattr(self.dtpr_adapter.dtpr.BDB, "index", [])) < 2:
            QMessageBox.information(
                self,
                self.tr("Load Beats First"),
                self.tr("Load a beat file before loading MIDI."),
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select a MIDI File"),
            "",
            "MIDI Files (*.mid *.midi);;All Files (*.*)",
        )
        if not file_path:
            return
        self.loaded_midi_path = file_path
        self.loaded_dtpr_path = None
        try:
            self._sync_dtpr_quantize_delta_from_grid_divisions()
            use_lm_quantize = bool(
                getattr(self, "chk_use_lm_quantize", None) is not None
                and self.chk_use_lm_quantize.isChecked()
            )
            self.dtpr_adapter.replace_notes_from_midi_file(
                file_path,
                use_language_model_quantization=use_lm_quantize,
            )
        except Exception as e:
            print(f"[DTPR midi load] failed: {e}", file=sys.stderr)
            traceback.print_exc()
            QMessageBox.critical(
                self,
                self.tr("MIDI Load Error"),
                self.tr("An error occurred while loading the MIDI file:\n{error}").format(error=e),
            )
            return

        log_path = str(Path(self.loaded_midi_path).with_suffix(".edit_log.jsonl"))
        self.edit_log_path = Path(log_path)
        self.current_process_id = None
        self.current_fun_id = None
        self.edit_logger = EditOperationLogger(log_path, auto_flush=True)
        self.edit_logger.append_session(meta={
            "midiPath": self.loaded_midi_path,
            "appVersion": APP_VERSION,
            "gridSec": 0.01,
            "minNoteLenSec": 0.02,
        })

        self.setup_pianoroll()
        self._sync_beats_from_dtpr()
        self.status_label.setText(
            self.tr("MIDI loaded: {filename}").format(
                filename=Path(file_path).name
            )
        )
        self.save_btn.setEnabled(True)
        self.save_score_midi_btn.setEnabled(True)
        self.export_qpr_btn.setEnabled(True)
        self._mark_saved_changes()

    def on_load_dtpr(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select a DTPR File"),
            "",
            "DTPR/Text Files (*.txt);;All Files (*.*)",
        )
        if not file_path:
            return
        try:
            self._load_dtpr_from_path(file_path, update_status=True)
        except Exception as e:
            print(f"[DTPR load] failed: {e}", file=sys.stderr)
            traceback.print_exc()
            QMessageBox.critical(
                self,
                self.tr("DTPR Load Error"),
                self.tr("An error occurred while loading the DTPR file:\n{error}").format(error=e),
            )

    # ---------- 音源 読み込み関連 ----------

    def _load_wav_common(self, which: int):
        """
        which = 1: 元音源, 2: 歌声分離音源
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select an Audio File"),
            "",
            "Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*.*)",
        )
        if not file_path:
            return

        try:
            self._load_audio_from_path(file_path, which=which, update_status=True)
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Audio Load Error"),
                self.tr("An error occurred while loading the audio file:\n{error}").format(error=e),
            )

    def on_load_wav1(self):
        self._load_wav_common(which=1)

    def on_load_wav2(self):
        self._load_wav_common(which=2)

    def _update_play_button_state(self):
        """
        再生ボタンを有効にする条件:
        - MIDI が読み込まれている
        - 音源が1つ以上読み込まれている
        """
        can_play_base = (
            self.dtpr_adapter is not None and
            (self.wav1_data is not None or self.wav2_data is not None) and
            self.audio_sr is not None
        )

        playing = self.playback_controller.is_playing
        paused = self.playback_controller.is_paused
        paused_time = self.playback_controller.paused_time

        self.play_btn.setEnabled((not playing) and can_play_base)
        self.pause_btn.setEnabled(playing)

        # ★Stopは「再生中」または「一時停止中（または途中位置）」なら有効
        self.stop_btn.setEnabled(playing or paused or (paused_time > 0.0))

    # ---- 出力チャンネル ----
    @staticmethod
    def _route_flags(route: str) -> tuple[bool, bool]:
        normalized = str(route or "none")
        return normalized in {"lr", "left"}, normalized in {"lr", "right"}

    def _sync_playback_route_state_from_controls(self) -> None:
        self.play_wav1_l, self.play_wav1_r = self._route_flags(
            self.cmb_wav1_route.currentData()
        )
        self.play_wav2_l, self.play_wav2_r = self._route_flags(
            self.cmb_wav2_route.currentData()
        )
        self.play_midi_l, self.play_midi_r = self._route_flags(
            self.cmb_midi_route.currentData()
        )

        self.play_wav1 = self.play_wav1_l or self.play_wav1_r
        self.play_wav2 = self.play_wav2_l or self.play_wav2_r
        self.play_midi = self.play_midi_l or self.play_midi_r

    def _midi_pan_value(self) -> int:
        if self.play_midi_l and not self.play_midi_r:
            return 0
        if self.play_midi_r and not self.play_midi_l:
            return 127
        return 64

    def on_audio_route_changed(self, _index: int):
        self._sync_playback_route_state_from_controls()
        self.playback_controller.update_midi_routing(
            self.play_midi,
            self._midi_pan_value(),
        )

    # ---- スライダ（0〜100、50を基準音量とするdBカーブ）----

    def on_slider_wav1_changed(self, value: int):
        self.volume_wav1 = _volume_slider_to_gain(value)

    def on_slider_wav2_changed(self, value: int):
        self.volume_wav2 = _volume_slider_to_gain(value)

    def on_slider_midi_changed(self, value: int):
        self.volume_midi = _volume_slider_to_gain(value)
        self.playback_controller.update_midi_volume(self.volume_midi)

    def on_metronome_click_toggled(self, checked: bool):
        self.play_metronome_click = bool(checked)
        self.playback_controller.update_metronome(
            self.play_metronome_click,
            self.volume_metronome,
        )

    def on_slider_metronome_changed(self, value: int):
        self.volume_metronome = _metronome_slider_to_gain(value)
        self.playback_controller.update_metronome(
            self.play_metronome_click,
            self.volume_metronome,
        )

    def on_speed_combo_changed(self, index: int):
        speed = float(self.cmb_speed.currentData() or 1.0)
        self.playback_speed = speed

    def on_midi_time_mode_toggled(self, checked: bool):
        self.midi_time_mode = "ideal" if checked else "real"

    def _build_mix_signature(self):
        return (
            id(self.wav1_data),
            id(self.wav2_data),
            bool(self.play_wav1),
            bool(self.play_wav1_l),
            bool(self.play_wav1_r),
            bool(self.play_wav2),
            bool(self.play_wav2_l),
            bool(self.play_wav2_r),
            round(float(self.volume_wav1), 4),
            round(float(self.volume_wav2), 4),
            round(float(self.wav1_playback_gain), 4),
            round(float(self.wav2_playback_gain), 4),
            int(self.audio_sr or 0),
        )

    def _compute_mtpr_max_tick(self) -> int:
        if self.dtpr_adapter is None:
            return int(TPQN * 8)
        return self.dtpr_adapter.get_mtpr_display_max_tick(default_beats=8)

    def _compute_mtpr_initial_px_per_tick(self) -> float:
        ptpr_px_per_sec = 1.0 / 0.01
        if self.scene is not None:
            try:
                ptpr_px_per_sec = 1.0 / float(self.scene.seconds_per_pixel)
            except Exception:
                ptpr_px_per_sec = 100.0

        if self.dtpr_adapter is None:
            return float(0.2)

        beat_sec = self.dtpr_adapter.get_median_beat_duration_sec()
        if beat_sec is None:
            return float(0.2)

        px_per_tick = float(ptpr_px_per_sec * beat_sec / float(TPQN))
        return float(max(0.005, min(1.0, px_per_tick)))

    def on_ptpr_scene_notes_changed_sync_mtpr(self):
        self._mark_unsaved_changes()
        if self.mtpr_scene is None:
            return
        try:
            self._sync_beats_from_dtpr()
            self._refresh_mtpr_grid_from_bdb()
            if self.scene is not None:
                self.scene.populate_notes()
            self.mtpr_scene.set_tick_range(self._compute_mtpr_max_tick())
            self.mtpr_scene.populate_notes()
        except Exception:
            pass

    def on_mtpr_scene_notes_changed_sync_ptpr(self):
        self._mark_unsaved_changes()
        if self.scene is None:
            return
        try:
            self._sync_beats_from_dtpr()
            self._refresh_mtpr_grid_from_bdb()
            if self.mtpr_scene is not None:
                self.mtpr_scene.set_tick_range(self._compute_mtpr_max_tick())
                self.mtpr_scene.populate_notes()
            self.scene.populate_notes()
            self._update_scene_time_range()
            self.on_notes_changed_for_confidence()
            self.on_notes_changed_refresh_candidates()
            if self.dev_table.isVisible():
                self.developer_panel_controller.update()
        except Exception:
            pass

    def _refresh_mtpr_grid_from_bdb(self) -> None:
        if self.mtpr_scene is None or self.dtpr_adapter is None:
            return
        self.mtpr_scene.set_bdb_grid(
            self.dtpr_adapter.dtpr.BDB,
            subdivisions_per_beat=self.mtpr_subdivisions_per_beat,
        )
        self.mtpr_scene.invalidate(self.mtpr_scene.sceneRect(), QGraphicsScene.BackgroundLayer)

    def _refresh_after_beat_edit(self) -> None:
        self._mark_unsaved_changes()
        self._sync_beats_from_dtpr()
        if self.scene is not None:
            self.scene.populate_notes()
        if self.mtpr_scene is not None:
            self._refresh_mtpr_grid_from_bdb()
            self.mtpr_scene.set_tick_range(self._compute_mtpr_max_tick())
            self._refresh_mtpr_feature_backgrounds_from_bdb()
            self.mtpr_scene.populate_notes()
        self._update_scene_time_range()
        self.on_notes_changed_for_confidence()
        self.on_notes_changed_refresh_candidates()
        if self.dev_table.isVisible():
            self.developer_panel_controller.update()

    def _sync_piano_roll_panel_refs(self) -> None:
        panel = self.piano_roll_panel
        self.scene = panel.scene
        self.view = panel.view
        self.keyboard = panel.keyboard
        self.mtpr_scene = panel.mtpr_scene
        self.mtpr_view = panel.mtpr_view
        self.mtpr_keyboard = panel.mtpr_keyboard
        self.waveform_widget = panel.waveform_widget
        self.pianoroll_splitter = panel.splitter

    def setup_pianoroll(self):
        if self.dtpr_adapter is None:
            return

        mtpr_max_tick = self._compute_mtpr_max_tick()
        mtpr_px_per_tick = self._compute_mtpr_initial_px_per_tick()
        container = self.piano_roll_panel.rebuild(
            edit_logger=self.edit_logger,
            dtpr_adapter=self.dtpr_adapter,
            beat_edit_mode=self.beat_edit_mode,
            subdivisions_per_beat=self.mtpr_subdivisions_per_beat,
            mtpr_max_tick=mtpr_max_tick,
            mtpr_px_per_tick=mtpr_px_per_tick,
            view_callbacks=self._piano_roll_view_callbacks(),
            vocal_audio=self.wav2_data,
            audio_sample_rate=self.audio_sr,
        )
        self._sync_piano_roll_panel_refs()
        self._update_scene_time_range()
        self.compute_and_set_background_features()

        try:
            self._ensure_mtpr_candidate_engine(reload_output=True)
        except Exception:
            pass

        self.editor_layout.removeWidget(self.left_widget)
        self.left_widget.setParent(None)
        self.left_widget = container
        self.editor_layout.insertWidget(0, self.left_widget, 3)

        self.save_btn.setEnabled(True)
        self.save_score_midi_btn.setEnabled(True)
        self.export_qpr_btn.setEnabled(True)
        self.status_label.setText(self.tr("MIDI loaded"))

        if self.dev_table.isVisible():
            self.developer_panel_controller.update()

        self.on_notes_changed_for_confidence()
        if getattr(self, "chk_show_spec_ptpr", None) is not None:
            self.on_chk_show_spec_ptpr_toggled(bool(self.chk_show_spec_ptpr.isChecked()))
        if getattr(self, "chk_show_spec_mtpr", None) is not None:
            self.on_chk_show_spec_mtpr_toggled(bool(self.chk_show_spec_mtpr.isChecked()))
        if getattr(self, "chk_show_chroma_ptpr", None) is not None:
            self.on_chk_show_chroma_ptpr_toggled(bool(self.chk_show_chroma_ptpr.isChecked()))
        if getattr(self, "chk_show_chroma_mtpr", None) is not None:
            self.on_chk_show_chroma_mtpr_toggled(bool(self.chk_show_chroma_mtpr.isChecked()))
        if getattr(self, "chk_show_pitch_prob_ptpr", None) is not None:
            self.on_chk_show_pitch_prob_ptpr_toggled(bool(self.chk_show_pitch_prob_ptpr.isChecked()))
        if getattr(self, "chk_show_pitch_prob_mtpr", None) is not None:
            self.on_chk_show_pitch_prob_mtpr_toggled(bool(self.chk_show_pitch_prob_mtpr.isChecked()))

        QTimer.singleShot(0, self.scroll_editor_to_start)

    def scroll_editor_to_start(self):
        self.piano_roll_panel.scroll_to_start()

    def on_undo_clicked(self):
        self.piano_roll_panel.undo()

    def on_redo_clicked(self):
        self.piano_roll_panel.redo()

    def _default_corr_output_path(self, suffix: str) -> str:
        base_name = "output_corr"
        source_path = self.loaded_midi_path or self.loaded_dtpr_path
        if source_path:
            try:
                path = Path(source_path)
                stem = path.stem
                if stem.endswith("_ipr"):
                    stem = stem[:-4]
                if stem.endswith("_dtpr"):
                    stem = stem[:-5]
                base_name = f"{stem}_corr"
                return str(path.with_name(f"{base_name}{suffix}"))
            except Exception:
                pass
        return f"{base_name}{suffix}"

    def on_save_midi(self):
        self._save_midi_with_time_mode(
            dialog_title=self.tr("Save Physical-time MIDI File"),
            default_suffix=".mid",
            time_mode="real",
        )

    def on_save_score_midi(self):
        self._save_midi_with_time_mode(
            dialog_title=self.tr("Save Metrical-time MIDI File"),
            default_suffix="_score.mid",
            time_mode="ideal",
        )

    def _save_midi_with_time_mode(
        self,
        *,
        dialog_title: str,
        default_suffix: str,
        time_mode: str,
    ) -> None:
        if self.dtpr_adapter is None or self.scene is None:
            return
        self.scene.sync_back_to_model()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            dialog_title,
            self._default_corr_output_path(default_suffix),
            "MIDI Files (*.mid)",
        )
        if not file_path:
            return
        path_obj = Path(file_path)
        if path_obj.suffix.lower() != ".mid":
            file_path = str(path_obj.with_suffix(".mid"))

        try:
            self.export_to_midi(file_path, time_mode=time_mode)
            self.status_label.setText(
                self.tr("Saved: {filename}").format(
                    filename=Path(file_path).name
                )
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Save Error"),
                self.tr("An error occurred while saving the MIDI file:\n{error}").format(error=e),
            )

    def on_export_qpr(self):
        if self.dtpr_adapter is None:
            return
        if self.scene is not None:
            self.scene.sync_back_to_model()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export DTPR File"),
            self._default_corr_output_path("_dtpr.txt"),
            "Text Files (*.txt);;All files (*.*)"
        )
        if not file_path:
            return
        path_obj = Path(file_path)
        if path_obj.suffix.lower() != ".txt":
            file_path = str(path_obj.with_suffix(".txt"))

        try:
            dtpr_class_to_dtpr_file(self.dtpr_adapter.dtpr, file_path)
            self.status_label.setText(
                self.tr("DTPR exported: {filename}").format(
                    filename=Path(file_path).name
                )
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Export Error"),
                self.tr("An error occurred while exporting the DTPR file:\n{error}").format(error=e),
            )

    def on_export_zip_package(self) -> bool:
        if self.dtpr_adapter is None:
            return False
        if not self.kkgk_session.package_path:
            QMessageBox.information(
                self,
                self.tr("Cannot Export KKGK"),
                self.tr("Only data loaded from a KKGK file can be exported as KKGK."),
            )
            return False
        if self.scene is not None:
            self.scene.sync_back_to_model()
        if self.mtpr_scene is not None:
            self.mtpr_scene.sync_back_to_model()

        default_path = str(Path(self.kkgk_session.package_path).with_suffix(".kkgk"))
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export KKGK File"),
            default_path,
            "KKGK Files (*.kkgk);;All Files (*.*)",
        )
        if not file_path:
            return False
        out_path = Path(file_path)
        if out_path.suffix.lower() != ".kkgk":
            out_path = out_path.with_suffix(".kkgk")

        dtpr_member = self.kkgk_session.output_dtpr_member(
            self.loaded_midi_path,
            self.loaded_dtpr_path,
        )

        try:
            if self.edit_log_path is None:
                raise ValueError(self.tr("The log file destination is not configured."))
            if not self.edit_log_path.exists():
                raise FileNotFoundError(
                    self.tr("Log file not found: {path}").format(
                        path=self.edit_log_path
                    )
                )
            log_member = self.kkgk_session.output_log_member(
                self.current_process_id,
                self.loaded_midi_path,
                self.loaded_dtpr_path,
            )
            with tempfile.TemporaryDirectory(prefix="kikigaki_zip_export_") as tmp:
                tmp_path = Path(tmp)
                dtpr_path = tmp_path / Path(dtpr_member).name

                dtpr_class_to_dtpr_file(self.dtpr_adapter.dtpr, str(dtpr_path))
                write_kkgk_package(
                    self.kkgk_session.package_path,
                    out_path,
                    dtpr_path,
                    dtpr_member,
                    self.edit_log_path,
                    log_member,
                )

                self.status_label.setText(
                    self.tr("KKGK exported: {filename}").format(
                        filename=out_path.name
                    )
                )
                self._mark_saved_changes()
                return True
        except Exception as e:
            print(f"[KKGK export] failed: {e}", file=sys.stderr)
            traceback.print_exc()
            QMessageBox.critical(
                self,
                self.tr("KKGK Export Error"),
                self.tr("An error occurred while exporting the KKGK file:\n{error}").format(
                    error=package_error_message(e, self.tr)
                ),
            )
            return False

    def export_to_midi(self, path: str, time_mode: str = "real"):
        if self.dtpr_adapter is None:
            return
        mode = str(time_mode).lower()
        if mode == "ideal":
            export_tick_notes_to_midi(
                self.dtpr_adapter.get_mtpr_notes_sorted(),
                path,
                ticks_per_beat=TPQN,
                bdb=self.dtpr_adapter.dtpr.BDB,
            )
            return
        export_notes_to_midi(
            self.dtpr_adapter.get_ptpr_notes_sorted(),
            path,
            time_mode=mode,
        )

    def _sync_dtpr_quantize_delta_from_grid_divisions(self) -> None:
        if self.dtpr_adapter is None:
            return
        divisions = max(1, int(getattr(self, "mtpr_grid_divisions", 4)))
        self.dtpr_adapter.set_quantization_grid(divisions)

    def _ensure_mtpr_candidate_engine(self, *, reload_output: bool = False):
        return self.candidate_controller.ensure_mtpr_engine(
            grid_divisions=int(getattr(self, "mtpr_grid_divisions", 4)),
            model_path=self.conf_ctrl.model_path,
            reload_output=reload_output,
        )

    def on_mtpr_grid_divisions_changed(self, _index: int):
        if not hasattr(self, "cmb_mtpr_grid_divisions") or self.cmb_mtpr_grid_divisions is None:
            return

        data = self.cmb_mtpr_grid_divisions.currentData()
        if data is None:
            try:
                data = int(str(self.cmb_mtpr_grid_divisions.currentText()))
            except Exception:
                return

        try:
            divisions = int(data)
        except Exception:
            return

        current_divisions = int(getattr(self, "mtpr_grid_divisions", 4))
        if not DTPRAdapter.is_grid_divisions_change_allowed(
            current_divisions,
            divisions,
        ):
            idx = self.cmb_mtpr_grid_divisions.findData(current_divisions)
            self.cmb_mtpr_grid_divisions.blockSignals(True)
            if idx >= 0:
                self.cmb_mtpr_grid_divisions.setCurrentIndex(idx)
            self.cmb_mtpr_grid_divisions.blockSignals(False)
            QMessageBox.information(
                self,
                self.tr("Cannot Change Divisions"),
                self.tr(
                    "The number of divisions cannot be reduced and can only be changed when the previous value is a divisor of the new value."
                ),
            )
            return

        self.mtpr_grid_divisions = int(divisions)
        self.mtpr_subdivisions_per_beat = max(1, int(divisions))
        self._sync_dtpr_quantize_delta_from_grid_divisions()

        if self.mtpr_scene is None:
            return

        self.mtpr_scene.subdivisions_per_beat = int(self.mtpr_subdivisions_per_beat)
        if self.dtpr_adapter is not None:
            try:
                self.mtpr_scene.set_bdb_grid(
                    self.dtpr_adapter.dtpr.BDB,
                    subdivisions_per_beat=self.mtpr_subdivisions_per_beat,
                )
            except Exception:
                pass
        try:
            engine = self._ensure_mtpr_candidate_engine(reload_output=False)
            if engine is not None:
                engine.set_separate_num(int(self.mtpr_grid_divisions))
        except Exception:
            pass
        self.mtpr_scene.update()
        self.refresh_candidates_from_selection()

    def on_zoom_in_clicked(self):
        self.piano_roll_panel.apply_x_zoom(zoom_in=True)

    def on_zoom_out_clicked(self):
        self.piano_roll_panel.apply_x_zoom(zoom_in=False)

    def on_vertical_zoom_in_clicked(self):
        self.piano_roll_panel.apply_y_zoom(zoom_in=True)

    def on_vertical_zoom_out_clicked(self):
        self.piano_roll_panel.apply_y_zoom(zoom_in=False)

    def on_scene_selection_changed(self):
        self.developer_panel_controller.sync_selection()
        self.refresh_candidates_from_selection()

    def on_keyboard_key_clicked(self, pitch: int):
        """
        鍵盤クリック時に呼ばれる。指定した pitch のノートを短く鳴らす。
        """
        if self.midi_player is not None:
            self.midi_player.play_note(pitch)

    def on_notes_audition_requested(self, note_ids: list[int]) -> None:
        if self.midi_player is None or self.dtpr_adapter is None:
            return
        if self.playback_controller.is_playing:
            return

        note_id = None
        for candidate_id in note_ids:
            note = self.dtpr_adapter.get_ptpr_note_by_id(int(candidate_id))
            if note is not None:
                note_id = int(candidate_id)
                break
        if note_id is None:
            return

        note = self.dtpr_adapter.get_ptpr_note_by_id(note_id)
        if note is None:
            return

        try:
            midi_gain = _volume_slider_to_gain(self.slider_midi.value())
            velocity = max(0, min(127, int(round(80 * midi_gain))))
            if velocity <= 0:
                return
            self.midi_player.play_note(
                int(note.pitch),
                velocity=velocity,
                duration_ms=250,
            )
        except Exception as e:
            print(f"MIDI audition error: {e}", file=sys.stderr)

    def on_open_settings_dialog(self):
        dialog = SettingsDialog(midi_program=self.midi_program, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.midi_program = dialog.selected_midi_program()
        if self.midi_player is not None:
            try:
                self.midi_player.set_program(self.midi_program)
            except Exception as e:
                print(f"MIDI program update error: {e}", file=sys.stderr)
        self.playback_controller.update_midi_program(self.midi_program)
        self.status_label.setText(
            self.tr("MIDI instrument: {program}").format(program=self.midi_program)
        )

    # ---------- 再生制御 ----------

    def _build_mix_audio(self) -> Optional[np.ndarray]:
        """
        読み込んだ 2 つの音源をミックスして stereo の float32 配列を返す。
        L/R 出力ボタンと音量スライダを反映。
        """
        tracks = []

        self._sync_playback_route_state_from_controls()

        if self.wav1_data is not None and self.play_wav1:
            tracks.append((
                self.wav1_data,
                self.volume_wav1 * self.wav1_playback_gain,
                self.play_wav1_l,
                self.play_wav1_r,
            ))
        if self.wav2_data is not None and self.play_wav2:
            tracks.append((
                self.wav2_data,
                self.volume_wav2 * self.wav2_playback_gain,
                self.play_wav2_l,
                self.play_wav2_r,
            ))

        return build_stereo_mix(tracks)

    def _max_wav_duration_seconds(self) -> float:
        return max_audio_duration_seconds(
            [self.wav1_data, self.wav2_data],
            self.audio_sr,
        )

    def _max_midi_duration_seconds(self) -> float:
        if self.dtpr_adapter is None:
            return 0.0
        return float(self.dtpr_adapter.get_ptpr_duration_sec(self.midi_time_mode))

    def _update_scene_time_range(self) -> None:
        if self.scene is None:
            return
        total = max(self._max_midi_duration_seconds(), self._max_wav_duration_seconds())
        if total <= 0.0:
            total = 10.0
        self.scene.set_total_time(total)

    def on_start_playback(self):
        if self.dtpr_adapter is None or self.audio_sr is None:
            QMessageBox.information(
                self,
                self.tr("Cannot Play"),
                self.tr("Load MIDI and audio before playback."),
            )
            return

        if self.playback_controller.is_playing:
            self.on_stop_playback()

        if self.scene is not None:
            self.scene.sync_back_to_model()

        self._sync_playback_route_state_from_controls()
        mix = self._build_mix_audio()
        beat_click_events = self._beat_click_events_for_playback()
        play_metronome = bool(self.play_metronome_click and beat_click_events)
        if mix is None and not self.play_midi and not play_metronome:
            QMessageBox.information(
                self,
                self.tr("Cannot Play"),
                self.tr("No audio or MIDI is selected for playback."),
            )
            return

        speed = float(getattr(self, "playback_speed", 1.0))
        audio_duration = self._max_wav_duration_seconds()
        midi_duration = float(
            self.dtpr_adapter.get_ptpr_duration_sec(self.midi_time_mode)
        )
        total_duration = max(audio_duration, midi_duration)

        result = self.playback_controller.start(
            mix_audio=mix,
            mix_signature=self._build_mix_signature(),
            sample_rate=self.audio_sr,
            notes=self.dtpr_adapter.get_ptpr_notes_sorted(),
            playback_speed=speed,
            total_duration=total_duration,
            parent=self,
            midi_out=self.midi_out,
            midi_port_name=self.midi_port_name,
            beat_events=beat_click_events,
            metronome_enabled=play_metronome,
            metronome_volume=self.volume_metronome,
            midi_time_mode=self.midi_time_mode,
            midi_program=self.midi_program,
            midi_enabled=self.play_midi,
            midi_pan_value=self._midi_pan_value(),
            midi_volume=self.volume_midi,
        )
        self._update_playback_time_label(result.start_offset)

        if result.reset_to_start:
            self.playback_slider.setValue(0)
            self._set_playhead_time_all(0.0)
            self._update_playback_time_label(0.0)

        if self.scene is not None:
            self.scene.playhead_draggable = False

        total_ms = max(0, int(result.total_duration * 1000))
        self.playback_slider.setRange(0, total_ms)
        if not result.was_paused:
            self.playback_slider.setValue(0)
        self.playback_slider.setEnabled(True)
        self.playback_slider.setFocusPolicy(Qt.NoFocus)

        self._update_play_button_state()
        self.status_label.setText(self.tr("Playing..."))

    def on_stop_playback(self):
        self.playback_controller.request_stop()
        if self.scene is not None:
            self.scene.playhead_draggable = True

        self._set_playhead_time_all(0.0)

        if getattr(self, "playback_slider", None) is not None:
            self.playback_slider.blockSignals(True)
            self.playback_slider.setValue(0)
            self.playback_slider.blockSignals(False)
        self._update_playback_time_label(0.0)

        self._update_play_button_state()

    def on_pause_playback(self):
        self.playback_controller.request_pause()

    def on_space_play_pause(self):
        if self.playback_controller.is_playing:
            self.on_pause_playback()
            return
        self.on_start_playback()

    def on_playback_finished(self, paused: bool, paused_time: float):
        self.playback_slider.setEnabled(True)

        if paused:
            self.playback_slider.setEnabled(True)
            self.status_label.setText(self.tr("Paused"))
            self._update_playback_time_label(paused_time)
            if self.scene is not None:
                self.scene.playhead_draggable = True
        else:
            self.playback_slider.setEnabled(False)
            self.playback_slider.setValue(0)
            self._set_playhead_time_all(0.0)
            self.status_label.setText(self.tr("Stop"))
            self._update_playback_time_label(0.0)
            if self.scene is not None:
                self.scene.playhead_draggable = True

        self._update_play_button_state()

    def on_playback_progress(self, t: float):
        """
        SyncAudioMidiPlayer からの progress シグナルで呼ばれる。
        t: 再生開始からの経過時間 [sec]
        """
        # スライダ更新（ミリ秒）
        if (
            self.playback_slider.isEnabled()
            and not self.playback_controller.is_slider_dragging
        ):
            value = int(t * 1000)
            # 範囲外にならないように clamp
            if value < 0:
                value = 0
            if value > self.playback_slider.maximum():
                value = self.playback_slider.maximum()
            self.playback_slider.setValue(value)

        # プレイヘッド更新
        self._set_playhead_time_all(t)

        self._update_playback_time_label(t)

        # 自動スクロール（ページ送り方式）
        if self.view is not None and self.scene is not None:
            x = t / self.scene.seconds_per_pixel  # プレイヘッドのシーン座標 X
            self._auto_scroll_view_for_x(self.view, x)
        mtpr_x = self._mtpr_playhead_x(t)
        if mtpr_x is not None:
            self._auto_scroll_view_for_x(self.mtpr_view, mtpr_x)

    def on_slider_pressed(self):
        self.playback_controller.begin_slider_drag()

    def _update_playback_time_label(self, t: float):
        if not hasattr(self, "playback_time_label") or self.playback_time_label is None:
            return
        cur = max(0.0, float(t))
        total = max(0.0, self.playback_controller.display_total_seconds)
        def fmt(sec: float) -> str:
            sec = max(0.0, float(sec))
            m = int(sec // 60)
            s = sec - (m * 60)
            return f"{m:02d}m{s:05.2f}s"
        self.playback_time_label.setText(f"{fmt(cur)} / {fmt(total)}")

    def _set_playhead_time_all(self, t: float):
        if self.scene is not None:
            self.scene.set_playhead_time(t)
        if self.mtpr_scene is not None:
            self.mtpr_scene.set_playhead_time(t)
        if self.waveform_widget is not None:
            self.waveform_widget.set_playhead_time(t)

    def _mtpr_playhead_x(self, t: float) -> Optional[float]:
        if self.mtpr_scene is None:
            return None
        abs_tick = self._mtpr_time_sec_to_abs_tick_continuous(float(t))
        return float(self.mtpr_scene.tick_to_x(abs_tick))

    @staticmethod
    def _auto_scroll_view_for_x(view: Optional[QGraphicsView], x: float) -> None:
        PianoRollPanel.auto_scroll_view_for_x(view, x)

    def on_playhead_dragged(self, t: float):
        if self.playback_controller.is_playing:
            return
        self.playback_controller.set_paused_position(float(t))
        paused_time = self.playback_controller.paused_time
        if getattr(self, "playback_slider", None) is not None:
            self.playback_slider.blockSignals(True)
            self.playback_slider.setValue(int(paused_time * 1000))
            self.playback_slider.blockSignals(False)
        self._set_playhead_time_all(paused_time)
        self._update_playback_time_label(paused_time)

    def on_slider_value_changed(self, value: int):
        if self.playback_controller.is_playing:
            return
        if not self.playback_controller.is_slider_dragging:
            return
        t = value / 1000.0

        # ドラッグ中は “見た目だけ” を更新（再生はまだ変えない）
        self._set_playhead_time_all(t)
        self._update_playback_time_label(t)

    def on_slider_released(self):
        if self.playback_controller.is_playing:
            self.playback_controller.end_slider_drag()
            return
        self.playback_controller.end_slider_drag()
        t = self.playback_slider.value() / 1000.0

        self.playback_controller.set_paused_position(t)
        self._set_playhead_time_all(t)
        self.scroll_view_to_time(t)

    def seek_restart(self, t: float):
        self.playback_controller.dispose_player()
        self.playback_controller.set_paused_position(max(0.0, float(t)))
        paused_time = self.playback_controller.paused_time
        self._set_playhead_time_all(paused_time)
        self.scroll_view_to_time(paused_time)
        self.on_start_playback()

    def scroll_view_to_time(self, t: float):
        self.piano_roll_panel.scroll_to_time(t)

    def on_chk_show_spec_ptpr_toggled(self, checked: bool):
        if self.scene is None:
            return
        self.scene.show_spectrogram = checked
        self.scene.invalidate(self.scene.sceneRect(), QGraphicsScene.BackgroundLayer)

    def on_chk_show_spec_mtpr_toggled(self, checked: bool):
        if self.mtpr_scene is None:
            return
        self.mtpr_scene.show_spectrogram = checked
        self.mtpr_scene.invalidate(self.mtpr_scene.sceneRect(), QGraphicsScene.BackgroundLayer)

    def on_chk_show_chroma_ptpr_toggled(self, checked: bool):
        if self.scene is None:
            return
        self.scene.show_chroma = checked
        self.scene.invalidate(self.scene.sceneRect(), QGraphicsScene.BackgroundLayer)

    def on_chk_show_chroma_mtpr_toggled(self, checked: bool):
        if self.mtpr_scene is None:
            return
        self.mtpr_scene.show_chroma = checked
        self.mtpr_scene.invalidate(self.mtpr_scene.sceneRect(), QGraphicsScene.BackgroundLayer)

    def on_chk_show_pitch_prob_ptpr_toggled(self, checked: bool):
        if self.scene is None:
            return
        self.scene.show_pitch_prob = checked
        if checked:
            self.update_pitch_prob_background()
        self.scene.invalidate(self.scene.sceneRect(), QGraphicsScene.BackgroundLayer)

    def on_chk_show_pitch_prob_mtpr_toggled(self, checked: bool):
        if self.mtpr_scene is None:
            return
        self.mtpr_scene.show_pitch_prob = checked
        if checked:
            self.update_pitch_prob_background()
        self.mtpr_scene.invalidate(self.mtpr_scene.sceneRect(), QGraphicsScene.BackgroundLayer)

    def _mtpr_time_sec_to_abs_tick_continuous(self, time_sec: float) -> float:
        if self.dtpr_adapter is None:
            return 0.0
        return self.dtpr_adapter.time_sec_to_abs_tick(time_sec)

    def _abs_tick_continuous_to_mtpr_time_sec(self, abs_tick: float) -> float:
        if self.dtpr_adapter is None:
            return 0.0
        return self.dtpr_adapter.abs_tick_to_time_sec(abs_tick)

    def on_view_x_zoom_changed(self, source_view: QGraphicsView, factor: float) -> None:
        self.piano_roll_panel.on_x_zoom_changed(source_view, factor)

    def on_view_y_zoom_changed(
        self,
        source_view: QGraphicsView,
        factor: float,
        center_scene_y: float,
    ) -> None:
        self.piano_roll_panel.on_y_zoom_changed(
            source_view,
            factor,
            center_scene_y,
        )

    def _refresh_mtpr_feature_backgrounds_from_bdb(self) -> None:
        self.feature_background_controller.refresh_mtpr_backgrounds(
            self.feature_cache,
            self.conf_ctrl.pitch_rest_logprobs,
        )

    def update_pitch_prob_background(self):
        self.feature_background_controller.apply_pitch_probability(
            self.conf_ctrl.pitch_rest_logprobs
        )

    def compute_and_set_background_features(self):
        if self.wav2_data is None or self.audio_sr is None or self.feature_cache is None:
            return
        self.feature_background_controller.apply_audio_features(self.feature_cache)

    def load_model_output_npy(self, path: str) -> None:
        self.conf_ctrl.load_model_output_npy(path)
        engine = self._ensure_mtpr_candidate_engine(reload_output=False)
        if engine is not None:
            engine.load_out_prob(path)
        self.update_pitch_prob_background()

    def on_load_confidence(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select confidence .npy",
            "",
            "NumPy files (*.npy);;All files (*.*)"
        )
        if not path:
            return

        try:
            self.load_model_output_npy(path)
            self.refresh_candidates_from_selection()
            self._update_confidence_controls_enabled()

            scene = getattr(self, "scene", None)
            if scene is not None:
                self.on_notes_changed_for_confidence()

        except Exception as e:
            print("Failed to load confidence:", e)
            self._update_confidence_controls_enabled()
    
    def _confidence_loaded(self) -> bool:
        """信頼度npyがロード済みか（UI制御用）。"""
        return self.conf_ctrl.has_confidence_data()

    def _update_confidence_controls_enabled(self) -> None:
        """信頼度UI（閾値・ジャンプ）の有効/無効を更新する。"""
        enabled = self._confidence_loaded()

        # ここで参照する前提：__init__ で self.conf_thr_spin / self.next_low_conf_btn を代入済み
        self.conf_thr_spin.setEnabled(enabled)
        self.next_low_conf_btn.setEnabled(False)

        # 任意：無効時の理由をツールチップで明示（好みで）
        if not enabled:
            msg = self.tr("Load a confidence file to enable this control.")
            self.conf_thr_spin.setToolTip(msg)
            self.next_low_conf_btn.setToolTip(msg)
        else:
            self.conf_thr_spin.setToolTip("")
            self.next_low_conf_btn.setToolTip("")

    def on_conf_threshold_changed(self, v: float):
        self.conf_ctrl.set_threshold(v)
        scene = getattr(self, "scene", None)
        if scene is None:
            return
        self.conf_ctrl.rebuild_low_conf_cache(scene)

    def on_next_low_conf(self):
        if not getattr(self, "enable_confidence_jump", False):
            return
        scene = getattr(self, "scene", None)
        view = getattr(self, "view", None)
        if scene is None or view is None:
            return
        ok = self.conf_ctrl.jump_next_low_conf(scene, view)
        if ok:
            # 選択の見た目を確実に更新したい場合
            scene.update()

    def on_notes_changed_for_confidence(self):
        # confidence が未ロードなら何もしない
        if self.conf_ctrl.series.data is None:
            return
        scene = getattr(self, "scene", None)
        if scene is None:
            return
        # 閾値キャッシュ更新 + 全ノート反映（最初は全更新で十分）
        self.conf_ctrl.set_threshold(self.conf_thr_spin.value())
        self.conf_ctrl.rebuild_low_conf_cache(scene)

        # 信頼度無効化
        if not getattr(self, "enable_confidence_coloring", False):
            return

        self.conf_ctrl.apply_to_all_items(scene)
        scene.update()

    def _candidate_confidence_data(self):
        series = getattr(self.conf_ctrl, "series", None)
        return None if series is None else series.data

    def _sync_candidate_pitch_modes(self) -> None:
        self.candidate_controller.set_pitch_modes(
            ptpr_enabled=bool(
                getattr(self, "chk_pitch_candidates", None)
                and self.chk_pitch_candidates.isChecked()
            ),
            mtpr_enabled=bool(
                getattr(self, "chk_mtpr_pitch_candidates", None)
                and self.chk_mtpr_pitch_candidates.isChecked()
            ),
        )

    def refresh_candidates_from_selection(self):
        self._sync_candidate_pitch_modes()
        self.candidate_controller.refresh_candidates_from_selection(
            grid_divisions=int(getattr(self, "mtpr_grid_divisions", 4)),
            model_path=self.conf_ctrl.model_path,
        )

    def on_candidate_mode_changed(self, idx: int):
        modes = ("off", "onset", "split")
        mode = modes[idx] if 0 <= idx < len(modes) else "split"
        self.candidate_controller.set_ptpr_mode(mode)
        self.refresh_candidates_from_selection()

    def on_mtpr_candidate_mode_changed(self, idx: int):
        modes = ("off", "onset", "offset", "split")
        mode = modes[idx] if 0 <= idx < len(modes) else "split"
        self.candidate_controller.set_mtpr_mode(mode)
        self.refresh_candidates_from_selection()

    def on_cycle_mtpr_candidate_mode(self):
        combo = getattr(self, "cmb_mtpr_candidates", None)
        if combo is None:
            return
        current = int(combo.currentIndex())
        next_index = {0: 1, 1: 2, 2: 3}.get(current, 1)
        if current == next_index:
            self.on_mtpr_candidate_mode_changed(next_index)
        else:
            combo.setCurrentIndex(next_index)

    def on_toggle_mtpr_pitch_candidates(self, checked: bool):
        self._sync_candidate_pitch_modes()
        if checked:
            self.refresh_candidates_from_selection()
        else:
            self.candidate_controller.clear_mtpr_pitch_candidates()

    def on_toggle_pitch_candidates(self, checked: bool):
        self._sync_candidate_pitch_modes()
        if checked:
            self.refresh_candidates_from_selection()
        else:
            self.candidate_controller.clear_pitch_candidates()

    def clear_onset_candidates(self):
        self.candidate_controller.clear_onset_candidates()

    def clear_pitch_candidates(self):
        self.candidate_controller.clear_pitch_candidates()

    def apply_candidate_by_number_key(self, number: int) -> bool:
        return self.candidate_controller.apply_candidate_by_number_key(number)

    def on_notes_changed_refresh_candidates(self):
        self._sync_candidate_pitch_modes()
        if not self.candidate_controller.has_active_candidates():
            return
        try:
            self.refresh_candidates_from_selection()
        except Exception:
            pass

    def _refresh_after_mtpr_candidate_apply(self, selected_ids: Optional[list[int]] = None):
        if self.mtpr_scene is not None:
            self.mtpr_scene.set_tick_range(self._compute_mtpr_max_tick())
            self.mtpr_scene.populate_notes()
        if self.scene is not None:
            self.scene.populate_notes()
            self._update_scene_time_range()
        if selected_ids and self.mtpr_scene is not None:
            self.mtpr_scene.blockSignals(True)
            self.mtpr_scene.clearSelection()
            for selected_id in selected_ids:
                item = self.mtpr_scene.note_items.get(int(selected_id))
                if item is not None:
                    item.setSelected(True)
            self.mtpr_scene.blockSignals(False)
        if self.dev_table.isVisible():
            self.developer_panel_controller.update()

    def on_load_beats_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open .beats file",
            "",
            "Beats (*.beats);;All Files (*)"
        )
        if not path:
            return

        try:
            events = load_beats_file(path)
        except Exception as e:
            print(f"[beats] failed to load: {e}")
            return

        self.beats_path = path
        self.dtpr_adapter = DTPRAdapter.from_beat_events(events)
        self._sync_dtpr_quantize_delta_from_grid_divisions()
        self.loaded_midi_path = None
        self.loaded_dtpr_path = None
        self.setup_pianoroll()
        self._sync_beats_from_dtpr()
        self.status_label.setText(
            self.tr("Beats loaded: {filename}").format(
                filename=Path(path).name
            )
        )
        self._mark_saved_changes()

    def on_toggle_beat_edit_mode(self, checked: bool):
        self.beat_edit_mode = bool(checked)
        self.beat_edit_controller.set_edit_mode(self.beat_edit_mode)
        if self.beat_edit_mode and getattr(self, "chk_show_beats", None) is not None:
            self.chk_show_beats.setChecked(True)
        if self.scene is not None:
            self.scene.beat_edit_mode = bool(self.beat_edit_mode)
            self.scene.note_editing_enabled = not bool(self.beat_edit_mode)
            self.scene.clearSelection()
            for item in list(self.scene.note_items.values()):
                item.setEnabled(not bool(self.beat_edit_mode))
        for control_name in ("cmb_candidates", "chk_pitch_candidates"):
            control = getattr(self, control_name, None)
            if control is not None:
                control.setEnabled(not bool(self.beat_edit_mode))
        if self.beat_edit_mode:
            self.clear_onset_candidates()
            self.clear_pitch_candidates()
        self.redraw_beats()

    def insert_beat_at_time(self, requested_time: float) -> None:
        self.beat_edit_controller.insert_beat_at_time(requested_time)

    def delete_selected_beats(self) -> None:
        self.beat_edit_controller.delete_selected_beats()

    def toggle_selected_beat_downbeat(self) -> None:
        self.beat_edit_controller.toggle_selected_downbeat()

    def on_toggle_beats(self, checked: bool):
        if checked:
            self.redraw_beats()
        else:
            self.clear_beats()

    def clear_beats(self):
        self.beat_edit_controller.clear_beats()

    def redraw_beats(self):
        visible = bool(
            getattr(self, "chk_show_beats", None) is None
            or self.chk_show_beats.isChecked()
        )
        self.beat_edit_controller.redraw_beats(
            visible=visible,
            beat_color=self.BEAT_COLOR,
            downbeat_color=self.DOWNBEAT_COLOR,
            beat_width=self.BEAT_WIDTH,
            downbeat_width=self.DOWNBEAT_WIDTH,
        )

    def _accept_close_event(self, event) -> None:
        try:
            if getattr(self, "playback_controller", None) is not None:
                self.playback_controller.request_stop()
        except Exception:
            pass
        event.accept()

    def closeEvent(self, event) -> None:
        if not self.has_unsaved_changes:
            self._accept_close_event(event)
            return

        if self.kkgk_session.can_export():
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(self.tr("Unsaved Changes"))
            box.setText(self.tr("There are unsaved changes."))
            box.setInformativeText(
                self.tr("Save as a KKGK file before closing?")
            )
            save_button = box.addButton(
                self.tr("Save and Close"), QMessageBox.AcceptRole
            )
            discard_button = box.addButton(
                self.tr("Close Without Saving"), QMessageBox.DestructiveRole
            )
            cancel_button = box.addButton(
                self.tr("Cancel"), QMessageBox.RejectRole
            )
            box.setDefaultButton(save_button)
            box.exec()

            clicked = box.clickedButton()
            if clicked is save_button:
                if self.on_export_zip_package():
                    self._accept_close_event(event)
                else:
                    event.ignore()
                return
            if clicked is discard_button:
                self._accept_close_event(event)
                return
            if clicked is cancel_button:
                event.ignore()
                return

        result = QMessageBox.warning(
            self,
            self.tr("Unsaved Changes"),
            self.tr("There are unsaved changes.\nClose without saving?"),
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if result == QMessageBox.Discard:
            self._accept_close_event(event)
        else:
            event.ignore()

def parse_startup_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--language",
        choices=["auto", "ja", "en"],
        default="en",
        help="UI language: auto, ja, or en.",
    )
    return parser.parse_known_args(argv)


def main():
    args, qt_args = parse_startup_args(sys.argv)
    app = QApplication(qt_args)
    app._kikigaki_translator = install_translator(app, args.language)
    app.aboutToQuit.connect(stop_fluidsynth_server)
    try:
        w = MainWindow()
    except Exception as e:
        msg = f"MIDI initialization error: {e}"
        print(msg, file=sys.stderr)
        QMessageBox.critical(None, "MIDI Initialization Error", msg)
        sys.exit(1)
    w.resize(1000, 650)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
