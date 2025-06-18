import sys
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QProgressBar, QMessageBox, QSizePolicy, QDial, QDialog, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QTimer
import re
from scipy.signal import butter, lfilter

class DeviceSelectDialog(QDialog):
    def __init__(self, input_devices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Input Device")
        self.setMinimumWidth(400)
        self.selected_index = None
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        # Use enumerate so idx matches the filtered input_devices list
        for idx, d in enumerate(input_devices):
            hostapi = sd.query_hostapis(d['hostapi'])['name']
            item = QListWidgetItem(f"{d['name']} [{hostapi}]")
            item.setData(Qt.UserRole, idx)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)
        select_btn = QPushButton("Select")
        select_btn.clicked.connect(self.select_device)
        layout.addWidget(select_btn)
        self.setLayout(layout)

    def select_device(self):
        selected = self.list_widget.currentItem()
        if selected:
            self.selected_index = selected.data(Qt.UserRole)
            self.accept()

class InputColumn(QWidget):
    def __init__(self, input_devices, idx):
        super().__init__()
        self.input_devices = input_devices
        self.selected_device_index = None  # No device selected by default

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setAlignment(Qt.AlignHCenter)

        self.label = QLabel(f"Input {idx+1}")
        self.layout.addWidget(self.label, alignment=Qt.AlignHCenter)

        # Sample rate label above the device selector
        self.sample_rate_label = QLabel("Sample Rate: 44100 Hz")
        self.sample_rate_label.setAlignment(Qt.AlignHCenter)
        self.layout.addWidget(self.sample_rate_label, alignment=Qt.AlignHCenter)

        # Device select button (replaces dropdown)
        self.device_select_btn = QPushButton("Select Input Device")
        self.device_select_btn.clicked.connect(self.open_device_dialog)
        self.layout.addWidget(self.device_select_btn)

        # Show selected device name (fixed width, elided if too long)
        self.selected_device_label = QLabel("No device selected")
        self.selected_device_label.setAlignment(Qt.AlignHCenter)
        self.selected_device_label.setFixedWidth(180)
        self.selected_device_label.setWordWrap(False)
        self.selected_device_label.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        self.layout.addWidget(self.selected_device_label)

        # --- Knobs for Bass, Treble, Equalizer ---
        knobs_layout = QHBoxLayout()
        # Bass knob
        bass_layout = QVBoxLayout()
        self.bass_knob = QDial()
        self.bass_knob.setMinimum(-10)
        self.bass_knob.setMaximum(10)
        self.bass_knob.setValue(0)
        self.bass_knob.setNotchesVisible(True)
        self.bass_knob.setFixedSize(48, 48)
        bass_label = QLabel("Bass")
        bass_label.setAlignment(Qt.AlignHCenter)
        bass_layout.addWidget(self.bass_knob, alignment=Qt.AlignHCenter)
        bass_layout.addWidget(bass_label)
        knobs_layout.addLayout(bass_layout)
        # Treble knob
        treble_layout = QVBoxLayout()
        self.treble_knob = QDial()
        self.treble_knob.setMinimum(-10)
        self.treble_knob.setMaximum(10)
        self.treble_knob.setValue(0)
        self.treble_knob.setNotchesVisible(True)
        self.treble_knob.setFixedSize(48, 48)
        treble_label = QLabel("Treble")
        treble_label.setAlignment(Qt.AlignHCenter)
        treble_layout.addWidget(self.treble_knob, alignment=Qt.AlignHCenter)
        treble_layout.addWidget(treble_label)
        knobs_layout.addLayout(treble_layout)
        # Equalizer knob
        eq_layout = QVBoxLayout()
        self.eq_knob = QDial()
        self.eq_knob.setMinimum(-10)
        self.eq_knob.setMaximum(10)
        self.eq_knob.setValue(0)
        self.eq_knob.setNotchesVisible(True)
        self.eq_knob.setFixedSize(48, 48)
        eq_label = QLabel("EQ")
        eq_label.setAlignment(Qt.AlignHCenter)
        eq_layout.addWidget(self.eq_knob, alignment=Qt.AlignHCenter)
        eq_layout.addWidget(eq_label)
        knobs_layout.addLayout(eq_layout)
        self.layout.addLayout(knobs_layout)
        # --- End knobs ---

        # Horizontal layout for visualizers (L and R), slider, and output buttons
        slider_vis_layout = QHBoxLayout()

        # Visualizers for left and right channels, side by side
        lr_vis_layout = QHBoxLayout()
        # Left channel
        left_vis_layout = QVBoxLayout()
        self.left_label = QLabel("L")
        self.left_label.setAlignment(Qt.AlignHCenter)
        left_vis_layout.addWidget(self.left_label)
        self.left_visualizer = QProgressBar()
        self.left_visualizer.setOrientation(Qt.Vertical)
        self.left_visualizer.setMinimum(0)
        self.left_visualizer.setMaximum(1000)
        self.left_visualizer.setTextVisible(False)
        self.left_visualizer.setFixedHeight(220)
        self.left_visualizer.setFixedWidth(28)
        self.left_visualizer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_vis_layout.addWidget(self.left_visualizer, alignment=Qt.AlignBottom)
        lr_vis_layout.addLayout(left_vis_layout)

        # Right channel
        right_vis_layout = QVBoxLayout()
        self.right_label = QLabel("R")
        self.right_label.setAlignment(Qt.AlignHCenter)
        right_vis_layout.addWidget(self.right_label)
        self.right_visualizer = QProgressBar()
        self.right_visualizer.setOrientation(Qt.Vertical)
        self.right_visualizer.setMinimum(0)
        self.right_visualizer.setMaximum(1000)
        self.right_visualizer.setTextVisible(False)
        self.right_visualizer.setFixedHeight(220)
        self.right_visualizer.setFixedWidth(28)
        self.right_visualizer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        right_vis_layout.addWidget(self.right_visualizer, alignment=Qt.AlignBottom)
        lr_vis_layout.addLayout(right_vis_layout)

        slider_vis_layout.addLayout(lr_vis_layout)

        # Slider
        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setFixedHeight(220)
        self.slider.setFixedWidth(40)
        slider_vis_layout.addWidget(self.slider, alignment=Qt.AlignBottom)

        # Output buttons A1-A6
        output_btns_layout = QVBoxLayout()
        self.output_buttons = []
        for i in range(6):
            btn = QPushButton(f"A{i+1}")
            btn.setCheckable(True)
            btn.setFixedWidth(36)
            btn.setFixedHeight(28)
            btn.setStyleSheet("QPushButton { margin: 2px; }")
            btn.setChecked(False)  # No connection by default
            output_btns_layout.addWidget(btn)
            self.output_buttons.append(btn)
        slider_vis_layout.addLayout(output_btns_layout)

        self.layout.addLayout(slider_vis_layout)

    def open_device_dialog(self):
        dialog = DeviceSelectDialog(self.input_devices, self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_index is not None:
            self.selected_device_index = dialog.selected_index
            self.selected_device_label.setText(self.get_selected_device_name())
            if self.device_selected_callback:
                self.device_selected_callback()

    def get_selected_device_name(self):
        if self.selected_device_index is None:
            return "No device selected"
        d = self.input_devices[self.selected_device_index]
        hostapi = sd.query_hostapis(d['hostapi'])['name']
        # Remove text in () or []
        name = re.sub(r"[\(\[].*?[\)\]]", "", d['name']).strip()
        return f"{name} {hostapi}"

class AudioMixerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4-Input Audio Mixer")
        self.resize(1200, 400)
        self.num_inputs = 4
        self.num_outputs = 6  # Now 6 outputs
        self.channels = 2  # Stereo
        self.samplerate = 48000  # or 44100, depending on your device
        self.blocksize = 4096

        # List input devices
        self.input_devices = [d for d in sd.query_devices() if d['max_input_channels'] >= self.channels]
        if len(self.input_devices) < self.num_inputs:
            QMessageBox.critical(self, "Error", "Not enough stereo input devices detected!")
            sys.exit(1)

        # List output devices
        self.output_devices = [d for d in sd.query_devices() if d['max_output_channels'] >= self.channels]
        if len(self.output_devices) < self.num_outputs:
            QMessageBox.critical(self, "Error", "Not enough stereo output devices detected!")
            sys.exit(1)

        # Main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Inputs in columns
        self.inputs_layout = QHBoxLayout()
        self.inputs_layout.setContentsMargins(10, 10, 0, 10)  # Add more left margin, less right
        self.inputs_layout.setSpacing(10)  # Optional: reduce spacing between columns
        self.input_columns = []
        for i in range(self.num_inputs):
            col = InputColumn(self.input_devices, i)
            self.inputs_layout.addWidget(col)
            self.input_columns.append(col)

        # Output selectors (vertical, to the right of inputs)
        self.output_selector_layout = QVBoxLayout()
        self.output_labels = []
        self.output_selectors = []
        for i in range(self.num_outputs):
            label = QLabel(f"Output {i+1}")
            self.output_labels.append(label)
            self.output_selector_layout.addWidget(label, alignment=Qt.AlignHCenter)
            selector = QComboBox()
            selector.setFixedWidth(300)
            selector.addItem("No device selected")  # Default: none selected
            for d in self.output_devices:
                hostapi = sd.query_hostapis(d['hostapi'])['name']
                selector.addItem(f"{d['name']} [{hostapi}]")
            selector.setCurrentIndex(0)  # No device selected by default
            self.output_selector_layout.addWidget(selector)
            self.output_selectors.append(selector)
        self.inputs_layout.addLayout(self.output_selector_layout)

        self.main_layout.addLayout(self.inputs_layout)

        # Remove start/stop buttons
        # self.buttons_layout = QHBoxLayout()
        # self.start_btn = QPushButton("Start Mixer")
        # self.stop_btn = QPushButton("Stop Mixer")
        # self.stop_btn.setEnabled(False)
        # self.buttons_layout.addWidget(self.start_btn)
        # self.buttons_layout.addWidget(self.stop_btn)
        # self.main_layout.addLayout(self.buttons_layout)

        # self.start_btn.clicked.connect(self.start_mixer)
        # self.stop_btn.clicked.connect(self.stop_mixer)

        self.input_buffers = [np.zeros((self.blocksize, self.channels), dtype=np.float32)
                              for _ in range(self.num_inputs)]
        self.running = False

        # Timer for mixing loop and visualizer update
        self.timer = QTimer()
        self.timer.timeout.connect(self.mix_and_route)

        # Connect input device selection to restart mixer
        for col in self.input_columns:
            col.device_selected_callback = self.reinitialize_mixer

        # Connect output device selection to restart mixer
        for selector in self.output_selectors:
            selector.currentIndexChanged.connect(self.reinitialize_mixer)

        # Start mixer automatically after a delay
        QTimer.singleShot(2000, self.start_mixer)

        # Remove the full screen button from the window
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)

    def reinitialize_mixer(self, *args, **kwargs):
        try:
            self.stop_mixer()
        except Exception as e:
            print("Error stopping mixer:", e)
        QTimer.singleShot(300, self.safe_start_mixer)

    def safe_start_mixer(self):
        try:
            self.start_mixer()
        except Exception as e:
            print("Error starting mixer:", e)
            QMessageBox.critical(self, "Audio Error", f"Could not start audio streams:\n{e}")

    def input_callback_factory(self, idx):
        def callback(indata, frames, time, status):
            self.input_buffers[idx] = indata.copy()
            # Stereo RMS for visualizers
            if indata.shape[1] == 2:
                left_rms = np.sqrt(np.mean(np.square(indata[:, 0])))
                right_rms = np.sqrt(np.mean(np.square(indata[:, 1])))
            else:
                left_rms = right_rms = np.sqrt(np.mean(np.square(indata)))
            # More steps: set maximum to 1000 for finer granularity
            sensitivity = 6000  # Adjust as needed for your signal
            self.input_columns[idx].left_visualizer.setMaximum(1000)
            self.input_columns[idx].right_visualizer.setMaximum(1000)
            self.input_columns[idx].left_visualizer.setValue(int(min(1000, left_rms * sensitivity)))
            self.input_columns[idx].right_visualizer.setValue(int(min(1000, right_rms * sensitivity)))
        return callback

    def start_mixer(self):
        device_indices = []
        for col in self.input_columns:
            # If no device selected, append None (will use silence)
            if col.selected_device_index is None:
                device_indices.append(None)
            else:
                device_indices.append(col.input_devices[col.selected_device_index]['index'])

        output_device_indices = []
        for selector in self.output_selectors:
            # If no device selected, append None (will not create output stream)
            if selector.currentIndex() == 0:
                output_device_indices.append(None)
            else:
                output_device_indices.append(self.output_devices[selector.currentIndex() - 1]['index'])

        try:
            self.inputs = []
            for i in range(self.num_inputs):
                if device_indices[i] is not None:
                    dev = self.input_devices[self.input_columns[i].selected_device_index]
                    print(f"Opening input {i}: {dev['name']} idx={dev['index']} channels={self.channels}")
                    self.inputs.append(
                        sd.InputStream(
                            device=dev['index'],
                            channels=self.channels,
                            samplerate=self.samplerate,
                            blocksize=self.blocksize,
                            callback=self.input_callback_factory(i)
                        )
                    )
                    self.inputs[-1].start()
                else:
                    # No device: fill buffer with zeros in callback
                    self.inputs.append(None)

            self.outputs = []
            for i in range(self.num_outputs):
                if output_device_indices[i] is not None:
                    self.outputs.append(
                        sd.OutputStream(
                            device=output_device_indices[i],
                            channels=self.channels,
                            samplerate=self.samplerate,
                            blocksize=self.blocksize,
                            callback=self.make_output_callback(i)
                        )
                    )
                    self.outputs[-1].start()
                else:
                    self.outputs.append(None)

            self.running = True
            self.timer.start(20)
            # self.start_btn.setEnabled(False)
            # self.stop_btn.setEnabled(True)
        except Exception as e:
            print("Audio stream error:", e)
            QMessageBox.critical(self, "Audio Error", f"Could not start audio streams:\n{e}")
            if hasattr(self, 'inputs'):
                for inp in self.inputs:
                    try:
                        if inp is not None:
                            inp.stop()
                            inp.close()
                    except Exception:
                        pass
            if hasattr(self, 'outputs'):
                for out in self.outputs:
                    try:
                        if out is not None:
                            out.stop()
                            out.close()
                    except Exception:
                        pass
            self.running = False
            # self.start_btn.setEnabled(True)
            # self.stop_btn.setEnabled(False)

    def stop_mixer(self):
        self.running = False
        self.timer.stop()
        for inp in self.inputs:
            inp.stop()
            inp.close()
        for out in self.outputs:
            out.stop()
            out.close()
        # self.start_btn.setEnabled(True)
        # self.stop_btn.setEnabled(False)
        for col in self.input_columns:
            col.left_visualizer.setValue(0)
            col.right_visualizer.setValue(0)

    def mix_and_route(self):
        self.mixed_buffers = []
        for out_idx in range(self.num_outputs):
            mix = np.zeros((self.blocksize, self.channels), dtype=np.float32)
            active = 0
            for in_idx, col in enumerate(self.input_columns):
                slider_val = col.slider.value() / 100.0
                lvl = np.power(slider_val, 4)
                if col.output_buttons[out_idx].isChecked():
                    # --- Apply Bass, Treble, EQ ---
                    buf = self.input_buffers[in_idx]
                    bass = col.bass_knob.value()
                    treble = col.treble_knob.value()
                    eq = col.eq_knob.value()
                    buf = apply_bass(buf, bass, self.samplerate)
                    buf = apply_treble(buf, treble, self.samplerate)
                    buf = apply_eq(buf, eq, self.samplerate)
                    # --- End filters ---
                    mix += buf * lvl
                    active += lvl
            if active > 0:
                mix /= active
            self.mixed_buffers.append(mix.astype(np.float32))
        levels = [col.slider.value() / 100.0 for col in self.input_columns]
        mixed = sum(buf * lvl for buf, lvl in zip(self.input_buffers, levels))
        if sum(levels) > 0:
            mixed /= sum(levels)
        else:
            mixed[:] = 0
        self.mixed_buffer = mixed.astype(np.float32)

    def make_output_callback(self, out_idx):
        def output_callback(outdata, frames, time, status):
            if hasattr(self, 'mixed_buffers') and len(self.mixed_buffers) > out_idx:
                outdata[:] = self.mixed_buffers[out_idx]
            else:
                outdata[:] = np.zeros((frames, self.channels), dtype=np.float32)
        return output_callback

    def restart_mixer_on_input_change(self):
        self.stop_mixer()
        self.start_mixer()

    def restart_mixer_on_output_change(self, idx):
        self.stop_mixer()
        self.start_mixer()

def apply_bass(audio, gain, samplerate):
    # gain: -10 to +10, 0 = no change
    if gain == 0:
        return audio
    b, a = butter(2, 200 / (samplerate / 2), btype='low')
    filtered = lfilter(b, a, audio, axis=0)
    return audio + (gain / 10.0) * filtered

def apply_treble(audio, gain, samplerate):
    if gain == 0:
        return audio
    b, a = butter(2, 4000 / (samplerate / 2), btype='high')
    filtered = lfilter(b, a, audio, axis=0)
    return audio + (gain / 10.0) * filtered

def apply_eq(audio, gain, samplerate):
    if gain == 0:
        return audio
    b, a = butter(2, [500/(samplerate/2), 2000/(samplerate/2)], btype='band')
    filtered = lfilter(b, a, audio, axis=0)
    return audio + (gain / 10.0) * filtered

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioMixerApp()
    window.show()
    sys.exit(app.exec_())

import sounddevice as sd

device_index = 2  # Replace with the index you get from your selection
channels = 2      # Or 1 for mono
samplerate = 44100

print(sd.query_devices(device_index))
try:
    with sd.InputStream(device=device_index, channels=channels, samplerate=samplerate) as stream:
        print("Stream opened successfully!")
except Exception as e:
    print("Error:", e)

import sounddevice as sd
for i, d in enumerate(sd.query_devices()):
    print(i, d['name'], d['max_input_channels'], d['max_output_channels'])