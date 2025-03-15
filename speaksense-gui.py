"""
SpeakSense GUI - A modern interface for the SpeakSense multimodal assistant system
"""

import sys
import os
import time
import threading
import queue
import logging
from typing import Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QComboBox, QSlider, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget, QFileDialog,
    QStatusBar, QFrame, QSplitter, QDialog, QFormLayout, QLineEdit,
    QMessageBox, QStyle, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor, QPalette, QTextCursor

from main import SpeakSense
from Live_transcription.Transcription import WhisperRealtimeTranscriber

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("speaksense_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SpeakSenseGUI")

# GUI color scheme
COLORS = {
    "primary": "#2c3e50",    # Dark blue-gray
    "secondary": "#3498db",   # Bright blue
    "accent": "#e74c3c",      # Red
    "success": "#2ecc71",     # Green
    "warning": "#f39c12",     # Orange
    "light": "#ecf0f1",       # Light gray
    "dark": "#2c3e50",        # Dark blue-gray
    "idle": "#95a5a6",        # Medium gray
    "listening": "#3498db",   # Blue
    "processing": "#f39c12",  # Orange
    "speaking": "#e74c3c",    # Red
}

class QClickableLabel(QLabel):
    """A clickable label widget"""
    clicked = pyqtSignal()
    
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

class LogHandler(logging.Handler):
    """Custom logging handler to redirect logs to the GUI"""
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
    def emit(self, record):
        msg = self.formatter.format(record)
        self.signal.emit(msg, record.levelno)

class VoiceActivityWidget(QWidget):
    """Widget displaying voice activity visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.energy_level = 0
        self.history = [0] * 100  # Keep 100 samples
        self.threshold = 0.02
        self.setMinimumHeight(60)
        
    def update_energy(self, energy_level, threshold=None):
        """Update the energy level to display"""
        self.energy_level = min(1.0, energy_level * 20)  # Scale for better visualization
        self.history.append(self.energy_level)
        self.history.pop(0)
        if threshold is not None:
            self.threshold = threshold
        self.update()
        
    def paintEvent(self, event):
        """Paint the voice activity visualization"""
        import math
        from PyQt5.QtGui import QPainter, QPen, QBrush
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(COLORS["light"]))
        
        # Draw threshold line
        threshold_y = self.height() * (1 - self.threshold * 20)
        painter.setPen(QPen(QColor(COLORS["warning"]), 1, Qt.DashLine))
        painter.drawLine(0, threshold_y, self.width(), threshold_y)
        
        # Draw energy history
        step_width = self.width() / (len(self.history) - 1)
        points = []
        
        for i, level in enumerate(self.history):
            x = i * step_width
            y = self.height() * (1 - level)
            points.append((x, y))
        
        # Draw the line
        if len(points) > 1:
            painter.setPen(QPen(QColor(COLORS["secondary"]), 2))
            for i in range(len(points) - 1):
                painter.drawLine(
                    points[i][0], points[i][1],
                    points[i+1][0], points[i+1][1]
                )
        
        # Draw the current level as a filled area
        if self.history[-1] > 0:
            painter.setPen(Qt.NoPen)
            
            # Set color based on whether it exceeds threshold
            if self.history[-1] > self.threshold * 20:
                painter.setBrush(QBrush(QColor(COLORS["success"])))
            else:
                painter.setBrush(QBrush(QColor(COLORS["idle"])))
                
            bar_width = 10
            bar_height = self.height() * self.history[-1]
            painter.drawRect(
                self.width() - bar_width - 5,
                self.height() - bar_height,
                bar_width,
                bar_height
            )

class StatusIndicator(QWidget):
    """Custom status indicator widget"""
    
    def __init__(self, text="Status", parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.indicator = QLabel()
        self.indicator.setFixedSize(12, 12)
        self.indicator.setStyleSheet(
            f"background-color: {COLORS['idle']}; border-radius: 6px;"
        )
        
        self.label = QLabel(text)
        self.status_text = QLabel("Idle")
        
        self.layout.addWidget(self.indicator)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.status_text)
        self.layout.addStretch()
        
    def set_status(self, status, color_key="idle"):
        """Update the status indicator"""
        self.status_text.setText(status)
        self.indicator.setStyleSheet(
            f"background-color: {COLORS[color_key]}; border-radius: 6px;"
        )

class SpeakSenseGUI(QMainWindow):
    """Main window for SpeakSense GUI application"""
    
    # Custom signals
    log_signal = pyqtSignal(str, int)  # For log messages: (message, level)
    status_signal = pyqtSignal(str)    # For status bar updates
    transcription_signal = pyqtSignal(str, float)  # Transcription text and confidence
    audio_energy_signal = pyqtSignal(float, float)  # Current energy level and threshold
    system_state_signal = pyqtSignal(dict)  # Various system state indicators
    
    def __init__(self):
        super().__init__()
        
        # Setup main window properties
        self.setWindowTitle("SpeakSense Assistant")
        self.setMinimumSize(900, 700)
        
        # Initialize system
        self.speaksense = None
        self.system_running = False
        self.energy_update_timer = QTimer()
        self.energy_update_timer.timeout.connect(self.update_energy_display)
        
        # Setup UI
        self.setup_ui()
        
        # Connect signals
        self.log_signal.connect(self.append_log)
        self.status_signal.connect(self.statusBar().showMessage)
        self.transcription_signal.connect(self.update_transcription)
        self.audio_energy_signal.connect(self.update_voice_activity)
        self.system_state_signal.connect(self.update_system_state)
        
        # Setup custom log handler
        log_handler = LogHandler(self.log_signal)
        log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(log_handler)
        
        # Initial status update
        self.status_signal.emit("Ready to start")
        
        # Create the SpeakSense system (but don't start it yet)
        self.init_speaksense()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create status bar
        self.statusBar().showMessage("Initializing...")
        
        # Create the main splitter
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.main_splitter)
        
        # Create top panel (controls and status)
        self.top_panel = QWidget()
        self.top_layout = QVBoxLayout(self.top_panel)
        
        # Control buttons
        self.controls_layout = QHBoxLayout()
        self.start_button = QPushButton("Start System")
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_button.clicked.connect(self.toggle_system)
        
        self.settings_button = QPushButton("Settings")
        self.settings_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.settings_button.clicked.connect(self.show_settings)
        
        self.controls_layout.addWidget(self.start_button)
        self.controls_layout.addWidget(self.settings_button)
        self.controls_layout.addStretch()
        self.top_layout.addLayout(self.controls_layout)
        
        # Status indicators
        self.status_box = QGroupBox("System Status")
        self.status_layout = QVBoxLayout(self.status_box)
        
        self.status_asd = StatusIndicator("Active Speaker Detection:")
        self.status_transcription = StatusIndicator("Transcription:")
        self.status_addressing = StatusIndicator("Addressing Detection:")
        self.status_ai = StatusIndicator("AI Response:")
        
        self.status_layout.addWidget(self.status_asd)
        self.status_layout.addWidget(self.status_transcription)
        self.status_layout.addWidget(self.status_addressing)
        self.status_layout.addWidget(self.status_ai)
        
        self.top_layout.addWidget(self.status_box)
        
        # Voice activity visualization
        self.activity_box = QGroupBox("Voice Activity")
        self.activity_layout = QVBoxLayout(self.activity_box)
        self.voice_activity = VoiceActivityWidget()
        self.activity_layout.addWidget(self.voice_activity)
        self.top_layout.addWidget(self.activity_box)
        
        # Add top panel to splitter
        self.main_splitter.addWidget(self.top_panel)
        
        # Create bottom panel with tabs
        self.bottom_panel = QTabWidget()
        
        # Conversation tab
        self.conversation_widget = QWidget()
        self.conversation_layout = QVBoxLayout(self.conversation_widget)
        
        self.transcription_label = QLabel("Current Transcription:")
        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setMinimumHeight(60)
        
        self.response_label = QLabel("AI Response:")
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setMinimumHeight(60)
        
        self.conversation_layout.addWidget(self.transcription_label)
        self.conversation_layout.addWidget(self.transcription_text)
        self.conversation_layout.addWidget(self.response_label)
        self.conversation_layout.addWidget(self.response_text)
        
        # Log tab
        self.log_widget = QWidget()
        self.log_layout = QVBoxLayout(self.log_widget)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_layout.addWidget(self.log_text)
        
        # Add tabs
        self.bottom_panel.addTab(self.conversation_widget, "Conversation")
        self.bottom_panel.addTab(self.log_widget, "System Log")
        
        # Add bottom panel to splitter
        self.main_splitter.addWidget(self.bottom_panel)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([350, 350])
    
    def init_speaksense(self):
        """Initialize the SpeakSense system (but don't start it)"""
        try:
            # Create the system object
            self.speaksense = SpeakSense()
            
            # Connect to monitoring signals if possible
            if hasattr(self.speaksense, 'state_manager'):
                self.speaksense.state_manager.add_state_change_listener(self.handle_state_change)
            
            # Success message
            logger.info("SpeakSense system initialized and ready")
            self.status_signal.emit("System initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SpeakSense: {e}")
            self.status_signal.emit("System initialization failed")
            QMessageBox.critical(
                self, 
                "Initialization Error", 
                f"Failed to initialize SpeakSense system: {str(e)}"
            )
    
    def toggle_system(self):
        """Start or stop the SpeakSense system"""
        if not self.system_running:
            self.start_system()
        else:
            self.stop_system()
    
    def start_system(self):
        """Start the SpeakSense system"""
        if not self.speaksense:
            self.init_speaksense()
            if not self.speaksense:
                return
        
        try:
            # Start the system in a separate thread to avoid blocking the GUI
            threading.Thread(
                target=self.speaksense.start, 
                daemon=True
            ).start()
            
            self.system_running = True
            self.start_button.setText("Stop System")
            self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
            self.settings_button.setEnabled(False)
            
            # Start the energy level update timer
            self.energy_update_timer.start(100)  # Update every 100ms
            
            logger.info("SpeakSense system started")
            self.status_signal.emit("System running")
            
            # Update status indicators
            self.status_asd.set_status("Active", "success")
            self.status_transcription.set_status("Listening", "success")
            self.status_addressing.set_status("Monitoring", "success")
            self.status_ai.set_status("Ready", "idle")
            
        except Exception as e:
            logger.error(f"Failed to start SpeakSense: {e}")
            self.status_signal.emit("Failed to start system")
            QMessageBox.critical(
                self, 
                "Startup Error", 
                f"Failed to start SpeakSense system: {str(e)}"
            )
    
    def stop_system(self):
        """Stop the SpeakSense system"""
        try:
            if self.speaksense:
                self.speaksense.stop()
            
            self.system_running = False
            self.start_button.setText("Start System")
            self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.settings_button.setEnabled(True)
            
            # Stop the energy level update timer
            self.energy_update_timer.stop()
            
            logger.info("SpeakSense system stopped")
            self.status_signal.emit("System stopped")
            
            # Update status indicators
            self.status_asd.set_status("Inactive", "idle")
            self.status_transcription.set_status("Stopped", "idle")
            self.status_addressing.set_status("Inactive", "idle")
            self.status_ai.set_status("Inactive", "idle")
            
        except Exception as e:
            logger.error(f"Error stopping SpeakSense: {e}")
            self.status_signal.emit("Error stopping system")
    
    def show_settings(self):
        """Show the settings dialog"""
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Apply settings
            self.apply_settings(dialog.get_settings())
    
    def apply_settings(self, settings):
        """Apply settings to the SpeakSense system"""
        logger.info(f"Applying settings: {settings}")
        
        # Apply settings if system exists
        if self.speaksense:
            # Example settings application - customize based on your system
            if hasattr(self.speaksense, 'addressing_worker') and hasattr(self.speaksense.addressing_worker, 'confidence_threshold'):
                self.speaksense.addressing_worker.confidence_threshold = settings['addressing_threshold']
            
            if hasattr(self.speaksense, 'speech_synthesizer') and hasattr(self.speaksense.speech_synthesizer, 'rate'):
                self.speaksense.speech_synthesizer.rate = settings['tts_rate']
            
            if hasattr(self.speaksense, 'speech_synthesizer') and hasattr(self.speaksense.speech_synthesizer, 'voice_id'):
                self.speaksense.speech_synthesizer.voice_id = settings['tts_voice']
    
    def update_energy_display(self):
        """Update the energy level display - called by timer"""
        if self.system_running and self.speaksense:
            # Access the audio energy level if available
            try:
                energy = 0.01  # Default low value
                threshold = 0.02  # Default threshold
                
                # Try to get the actual values from system
                if (hasattr(self.speaksense, 'transcription_worker') and 
                    hasattr(self.speaksense.transcription_worker, 'transcriber')):
                    
                    transcriber = self.speaksense.transcription_worker.transcriber
                    
                    # Get current audio buffer if available
                    if hasattr(transcriber, 'audio_buffer') and len(transcriber.audio_buffer) > 0:
                        import numpy as np
                        energy = np.abs(transcriber.audio_buffer[-1000:]).mean()
                    
                    # Get current threshold if available
                    if hasattr(transcriber, 'silence_threshold'):
                        threshold = transcriber.silence_threshold
                
                # Emit the signal to update the display
                self.audio_energy_signal.emit(energy, threshold)
                
            except Exception as e:
                logger.debug(f"Error updating energy display: {e}")
    
    def update_voice_activity(self, energy, threshold):
        """Update the voice activity widget"""
        self.voice_activity.update_energy(energy, threshold)
    
    def update_transcription(self, text, confidence):
        """Update the transcription display"""
        self.transcription_text.setText(f"{text} (Confidence: {confidence:.2f})")
    
    def update_system_state(self, state):
        """Update system state indicators based on state dictionary"""
        # Example state keys: 'asd_active', 'ai_speaking', 'addressing_detected'
        
        if 'asd_active' in state:
            self.status_asd.set_status(
                "Looking at Camera" if state['asd_active'] else "No Face Detected",
                "success" if state['asd_active'] else "idle"
            )
        
        if 'ai_speaking' in state:
            self.status_ai.set_status(
                "Speaking" if state['ai_speaking'] else "Listening",
                "speaking" if state['ai_speaking'] else "idle"
            )
        
        if 'addressing_detected' in state:
            self.status_addressing.set_status(
                "Addressing Assistant" if state['addressing_detected'] else "Monitoring",
                "success" if state['addressing_detected'] else "idle"
            )
    
    def handle_state_change(self, state_name, value):
        """Handle state changes from the SpeakSense system"""
        logger.debug(f"State change: {state_name} = {value}")
        
        # Update state based on what changed
        state_update = {}
        
        if state_name == "ai_speaking":
            state_update['ai_speaking'] = value
            
            # Also update the response text if we just finished speaking
            if not value and hasattr(self.speaksense, 'response_generator'):
                # Try to get the last response
                try:
                    last_response = self.speaksense.response_generator.last_response
                    if last_response:
                        self.response_text.setText(last_response)
                except:
                    pass
        
        # Emit signal with state updates
        if state_update:
            self.system_state_signal.emit(state_update)
    
    @pyqtSlot(str, int)
    def append_log(self, message, level):
        """Append a message to the log display with appropriate formatting"""
        # Define colors for different log levels
        colors = {
            logging.DEBUG: "#6c757d",    # Gray
            logging.INFO: "#2c3e50",     # Dark blue
            logging.WARNING: "#f39c12",  # Orange
            logging.ERROR: "#e74c3c",    # Red
            logging.CRITICAL: "#c0392b"  # Dark red
        }
        
        # Format with color
        color = colors.get(level, "#2c3e50")
        formatted_msg = f'<span style="color:{color};">{message}</span>'
        
        # Append to log
        self.log_text.append(formatted_msg)
        
        # Scroll to the bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.system_running:
            # Ask for confirmation before closing
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                "SpeakSense is still running. Stop and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Stop the system
                self.stop_system()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

class SettingsDialog(QDialog):
    """Settings dialog for SpeakSense"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SpeakSense Settings")
        self.setMinimumWidth(400)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the settings dialog UI"""
        self.layout = QVBoxLayout(self)
        
        # Create tabs for different settings categories
        self.tabs = QTabWidget()
        self.general_tab = QWidget()
        self.audio_tab = QWidget()
        self.transcription_tab = QWidget()
        self.ai_tab = QWidget()
        
        # Set up general settings
        self.setup_general_tab()
        
        # Set up audio settings
        self.setup_audio_tab()
        
        # Set up transcription settings
        self.setup_transcription_tab()
        
        # Set up AI settings
        self.setup_ai_tab()
        
        # Add tabs to widget
        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.audio_tab, "Audio")
        self.tabs.addTab(self.transcription_tab, "Transcription")
        self.tabs.addTab(self.ai_tab, "AI Response")
        
        self.layout.addWidget(self.tabs)
        
        # Add buttons
        self.button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.cancel_button)
        
        self.layout.addLayout(self.button_layout)
    
    def setup_general_tab(self):
        """Set up the general settings tab"""
        layout = QFormLayout(self.general_tab)
        
        # Addressing confidence threshold
        self.addressing_threshold = QDoubleSpinBox()
        self.addressing_threshold.setRange(0.1, 0.9)
        self.addressing_threshold.setSingleStep(0.05)
        self.addressing_threshold.setValue(0.6)
        layout.addRow("Addressing confidence threshold:", self.addressing_threshold)
        
        # Silence threshold
        self.silence_threshold = QDoubleSpinBox()
        self.silence_threshold.setRange(0.001, 0.1)
        self.silence_threshold.setSingleStep(0.005)
        self.silence_threshold.setValue(0.01)
        layout.addRow("Silence threshold:", self.silence_threshold)
        
        # Auto-calibration
        self.auto_calibrate = QCheckBox("Auto-calibrate silence threshold")
        self.auto_calibrate.setChecked(True)
        layout.addRow("", self.auto_calibrate)
    
    def setup_audio_tab(self):
        """Set up the audio settings tab"""
        layout = QFormLayout(self.audio_tab)
        
        # Microphone selection
        self.microphone = QComboBox()
        # Populate with available microphones
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    self.microphone.addItem(
                        f"{device_info['name']} (Index: {i})",
                        i  # Store device index as item data
                    )
            p.terminate()
        except:
            self.microphone.addItem("Default Microphone", -1)
        
        layout.addRow("Microphone:", self.microphone)
        
        # Sample rate
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(["16000 Hz", "22050 Hz", "44100 Hz"])
        self.sample_rate.setCurrentText("16000 Hz")
        layout.addRow("Sample rate:", self.sample_rate)
        
        # Audio feedback prevention
        self.prevent_feedback = QCheckBox("Enable advanced feedback prevention")
        self.prevent_feedback.setChecked(True)
        layout.addRow("", self.prevent_feedback)
    
    def setup_transcription_tab(self):
        """Set up the transcription settings tab"""
        layout = QFormLayout(self.transcription_tab)
        
        # Model selection
        self.model = QComboBox()
        self.model.addItems([
            "openai/whisper-tiny",
            "openai/whisper-base",
            "openai/whisper-small",
            "openai/whisper-medium",
            "openai/whisper-large"
        ])
        layout.addRow("Whisper model:", self.model)
        
        # Model optimization
        self.optimize_model = QCheckBox("Use half precision (faster on CUDA)")
        self.optimize_model.setChecked(True)
        layout.addRow("", self.optimize_model)
        
        # Minimum confidence
        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0.0, 1.0)
        self.min_confidence.setSingleStep(0.05)
        self.min_confidence.setValue(0.4)
        layout.addRow("Minimum confidence:", self.min_confidence)
    
    def setup_ai_tab(self):
        """Set up the AI settings tab"""
        layout = QFormLayout(self.ai_tab)
        
        # TTS voice selection
        self.tts_voice = QComboBox()
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            for i, voice in enumerate(voices):
                self.tts_voice.addItem(f"{voice.name} ({voice.id})", i)
            engine.stop()
        except:
            self.tts_voice.addItem("Default Voice", 0)
            self.tts_voice.addItem("Alternative Voice", 1)
        
        layout.addRow("Text-to-Speech voice:", self.tts_voice)
        
        # TTS rate
        self.tts_rate = QSlider(Qt.Horizontal)
        self.tts_rate.setRange(50, 300)
        self.tts_rate.setValue(140)
        self.tts_rate.setTickPosition(QSlider.TicksBelow)
        self.tts_rate.setTickInterval(50)
        
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Slow"))
        rate_layout.addWidget(self.tts_rate)
        rate_layout.addWidget(QLabel("Fast"))
        
        layout.addRow("Speech rate:", rate_layout)
        
        # System message
        self.system_message = QTextEdit()
        self.system_message.setPlainText(
            "You are an AI model who gives replies to crippled conversation "
            "try your best to figure out what the user meant from the cut off "
            "or maybe not so cut off user query, reply in 2 sentences"
        )
        self.system_message.setMaximumHeight(100)
        layout.addRow("System message:", self.system_message)
    
    def get_settings(self):
        """Get the current settings as a dictionary"""
        settings = {
            # General settings
            'addressing_threshold': self.addressing_threshold.value(),
            'silence_threshold': self.silence_threshold.value(),
            'auto_calibrate': self.auto_calibrate.isChecked(),
            
            # Audio settings
            'microphone_index': self.microphone.currentData(),
            'sample_rate': int(self.sample_rate.currentText().split()[0]),
            'prevent_feedback': self.prevent_feedback.isChecked(),
            
            # Transcription settings
            'model': self.model.currentText(),
            'optimize_model': self.optimize_model.isChecked(),
            'min_confidence': self.min_confidence.value(),
            
            # AI settings
            'tts_voice': self.tts_voice.currentData(),
            'tts_rate': self.tts_rate.value(),
            'system_message': self.system_message.toPlainText()
        }
        
        return settings

def main():
    """Main function to start the SpeakSense GUI"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Apply custom palette for a modern look
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["light"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["dark"]))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS["dark"]))
    palette.setColor(QPalette.Text, QColor(COLORS["dark"]))
    palette.setColor(QPalette.Button, QColor(COLORS["light"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["dark"]))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(COLORS["secondary"]))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    # Create and show the main window
    window = SpeakSenseGUI()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
