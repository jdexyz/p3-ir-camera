# P3 IR Camera

Python driver and viewer for P3-series USB thermal cameras.  Improved? with lock-in thermography function.  See LOCK-IN.md

![P3 Viewer - Keyboard](screenshots/jvdillon-keyboard.png)
![P3 Viewer - Chip](screenshots/huberbenno-chip.png)

Images courtesy of [jvdillon](https://github.com/jvdillon) and
[huberbenno](https://github.com/huberbenno)
([PR#11](https://github.com/jvdillon/p3-ir-camera/pull/11)).

![ESP32](screenshots/esp32-lockin.png)

**Devices**:

- P1: VID=0x3474, PID=0x45C2, 160×120 native resolution
- P3: VID=0x3474, PID=0x45A2, 256×192 native resolution

> **Disclaimer**: This is an independent open-source project. It is not
> affiliated with, endorsed by, or connected to any camera manufacturer.
> Protocol details were determined through USB traffic analysis and
> experimentation.

## Features

- USB driver for frame capture and device control
- Real-time thermal viewer with multiple colormaps
- Temperature measurement at cursor position
- Temporal noise reduction and digital detail enhancement
- Multiple AGC modes (factory hardware AGC, temporal percentile, fixed range)
- Shutter/NUC calibration control
- High/Low gain mode switching
- Rudimentary lock-in thermography for finding very small temperature changes

## Installation

```bash
git clone https://github.com/jvdillon/p3-ir-camera
cd p3-ir-camera
pip install -e .
```

### USB Permissions (Linux)

Create a udev rule to allow non-root access:

```bash
sudo tee /etc/udev/rules.d/99-p3-ir.rules << EOF
# P1 camera
SUBSYSTEM=="usb", ATTR{idVendor}=="3474", ATTR{idProduct}=="45c2", MODE="0666"
# P3 camera
SUBSYSTEM=="usb", ATTR{idVendor}=="3474", ATTR{idProduct}=="45a2", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### USB Driver (Windows)

pyusb requires a libusb-compatible driver. Use [Zadig](https://zadig.akeo.ie/):

1. Download and run Zadig
2. Options → List All Devices
3. Select the camera (VID 3474, PID 45C2 for P1 or 45A2 for P3)
4. Select **WinUSB** driver
5. Click "Replace Driver" (or "Reinstall Driver" if WinUSB is already bound)

Zadig installs the *driver*. pyusb additionally needs the libusb *library*, which
Zadig does not provide -- without it every call fails with
`usb.core.NoBackendError: No backend available`:

```bash
pip install libusb-package
```

Then copy `libusb-1.0.dll` from the installed `libusb_package/` directory into any
directory on your `PATH` (e.g. `%USERPROFILE%\.local\bin`). pyusb locates it via
`ctypes.util.find_library`, which on Windows searches `PATH` only -- not the current
directory and not site-packages.

## Usage

### GUI

```bash
p3-gui                                  # or: python p3_gui.py
p3-gui --range 20 350 --gain low        # start on a fixed absolute scale
p3-gui --range 20 350 --log --gain low  # ...mapped logarithmically
```

A windowed front-end for sessions where recording has to start and stop on cue.
Start/stop recording with a button, with elapsed time, frame count and raw size
shown live; set the output folder and name before you start; change the
temperature scale, gain, emissivity and colormap while streaming; trigger a
shutter/NUC or grab a snapshot.

It drives the same `P3Viewer` as the OpenCV viewer, so the image and the
recorded files are identical either way. Needs only tkinter (stdlib) and
Pillow (already required by matplotlib).

#### Reconnecting

The USB link drops for two ordinary reasons: the cable is pulled, or the machine
sleeps and the handle is invalidated on resume. The GUI recovers from both without
being restarted -- the **Camera link** panel shows `Connected` / `Reconnecting`, and
the capture thread retries until the device comes back.

Sleep is the awkward case, because reads may neither fail nor deliver: the image
just stops updating. A watchdog treats a gap of more than 14 s as a dead link and
forces a reconnect, so a frozen image becomes a visible `Reconnecting` state rather
than a stale picture that looks live. **Reconnect** forces the same cycle by hand.

A recording running when the link drops is finalized rather than abandoned, so its
`.mp4`, `.raw` and `.json` stay valid and hold every frame captured up to the
disconnect. Recording does not resume by itself -- start it again once the camera
reconnects.

#### View

**View** switches the preview between **Both** (default), **Thermal** and
**Visible**. Both stacks the feeds vertically, thermal above visible, at a common
width -- both cameras are landscape, so side by side would give a very wide, short
image that wastes most of a window. If one camera is unavailable the other is shown
alone rather than the preview going blank.

Showing and recording the webcam are separate: the device is opened whenever it is
being shown or recorded, and released when neither applies, so its light goes out
and other applications can use it.

#### Visible camera

Tick **Record visible alongside thermal** to record a USB (UVC) webcam at the same
time. The device list is probed at startup and labelled by resolution, since OpenCV
exposes no device names and its indices change when devices are replugged. The
highest-resolution device is selected by default, which picks a dedicated capture
camera over a built-in webcam (the built-in one always takes index 0). `--uvc INDEX`
preselects a specific one.

The visible stream is written at the webcam's own frame rate, and its frame count,
resolution and start time are added to the sidecar under `visible`. The two cameras
are independent devices with their own clocks, so their frames do not correspond
one-for-one; align them with `started_at` and `measured_fps` rather than by frame
index.

#### Focus

The **Autofocus** checkbox and **Focus** slider drive a webcam with a motorised lens;
the slider is greyed out while autofocus is on, and the value shown is the position
the lens reported back rather than the one requested, since the device quantises and
clamps what it is given.

This needs the DirectShow backend on Windows. Media Foundation, OpenCV's default
there, accepts focus writes, returns success and silently ignores them. If DirectShow
cannot open a device the code falls back to the default backend, so capture still
works -- without focus control. A camera with a fixed lens reports `-1` and the
controls do nothing.

#### Sound

**Record sound** captures a microphone alongside the video and muxes it into
`combined.mp4`, so that file plays with audio. The microphone list defaults to one
whose name looks like a camera's own (e.g. `Microphone (UGREEN Camera 2K)`) rather
than the machine's built-in array.

Two corrections keep it aligned. A `VideoWriter` must be told a frame rate before the
first frame exists, so the container's declared rate is a guess; the real rate is only
known once capture ends, and the timestamps are rescaled to it at mux time. Audio also
starts on the button press while the video writer cannot open until a frame has sized
it, so that lead is trimmed off the sound. A measured session showed a 5 ms lead and
0.1% rate error, both corrected.

OpenCV cannot record audio, so this needs `sounddevice` for capture and ffmpeg for
muxing; `imageio-ffmpeg` supplies a bundled ffmpeg binary, and a system ffmpeg is used
if present. If either is missing the capture still runs -- the sound is simply left as
`audio.wav` beside the video, and the sidecar says so. Muxing uses `-shortest`, so the
combined file is as long as the shorter of the two streams.

### Replay

```bash
p3-replay                 # pick a session folder
p3-replay path/to/run01   # or open one directly
```

Also reachable from the GUI's **Replay a recording...** button. Plays a session back
with sound, and **clicking the thermal image reads the temperature at that point**.

The thermal pane is rendered from `thermal.raw`, not from the recorded video: the
video holds 8-bit colour-mapped pixels, while the raw stream holds the sensor's own
counts. Only the raw data can answer what temperature a point was, and only it maps
cleanly back to a sensor pixel, so the probe is exact rather than inferred from
colours. The visible pane and the sound come from `combined.mp4`.

Playback is driven from the wall clock rather than by counting frames, so the picture
keeps pace with the sound instead of drifting behind it.

### Viewer

```bash
# Use P3 camera (default, 256×192)
p3-viewer

# Use P1 camera (160×120)
p3-viewer --model=p1

# Use P3 camera explicitly
p3-viewer --model=p3

# Lock-in thermography - press 'l' once viewer is open
p3-viewer --frequency 0.1 --integration 120
```

### Absolute temperature scale

By default the display autoscales, so colors mean different temperatures from frame
to frame. `--range` pins the mapping to a fixed absolute span (FIXED_RANGE AGC), which
stays constant over time and labels the colorbar with those bounds:

```bash
# Fixed 20-350 C scale; LOW gain is required above 150 C
p3-viewer --range 20 350 --gain low
```

`--gain` accepts `low` (0-550 C) or `high` (-20-150 C). A scene that only spans part of
a wide fixed range will look dark -- that is the scale working correctly, not a fault.

#### Logarithmic scale

A linear scale wide enough for a hot process crushes everything near ambient into the
bottom few counts. `--log` keeps the same absolute, time-invariant bounds but maps them
logarithmically, so room temperature and a hot target are both readable at once:

```bash
p3-viewer --range 20 350 --log --gain low
p3-viewer --range 20 350 --log --log-strength 150 --gain low
```

Display value for a 20-350 C scale:

| Scene | Linear | `--log` (k=50) | `--log-strength 150` |
| --- | --- | --- | --- |
| 30 C | 7/255 | 59/255 | 87/255 |
| 60 C | 30/255 | 126/255 | 150/255 |
| 150 C | 100/255 | 196/255 | 208/255 |
| 350 C | 254/255 | 254/255 | 254/255 |

The logarithmic mapping is the **default** for `--range`; pass `--linear` for the
plain linear mapping.

`--log-strength` (default 50) sets the curve: lower approaches linear, higher lifts the
cool end further. It applies only in log mode -- in the GUI the slider is greyed out
in the linear modes, since a live-looking control that changes nothing reads as a
bug. Select **Log range** and the strength shown takes effect immediately. The colorbar tick labels follow the same curve, so temperatures still
read off correctly -- the ticks are just no longer evenly spaced in temperature.

The curve is applied to position within the range, not to absolute temperature: a true
log of Kelvin is nearly linear over any range this camera can see, so it would do
almost nothing.

Logarithmic mapping affects the **display only**. The recorded `.raw` stream is
unchanged sensor data either way.

### Recording

The GUI writes one folder per session, named after the recording, so the files that
only make sense together stay together:

```
run01/
  combined.mp4    Both feeds stacked in one file, with sound
  thermal.mp4     Thermal feed alone -- colormap, overlays, colorbar
  thermal.tz      16-bit frames, pre-TNR sensor counts, block-compressed
  thermal.json    Frame counts, start times, measured rates, gain/AGC, emissivity
  audio.wav       Only if muxing was unavailable
```

#### Thermal storage

A plain 16-bit stream is about 150 MB per minute. Consecutive thermal frames are
nearly identical, so `thermal.tz` stores each frame as its difference from the one
before and compresses that with zstd -- **8.3x smaller on a measured session, and
exactly lossless**, taking 147 MB/min down to 18 MB/min. Encoding runs about 10x
faster than real time, so it keeps up with capture comfortably.

Compression usually costs random access, which replay needs for scrubbing. Frames
are therefore grouped into one-second blocks with an index, so seeking decompresses
a single block (~11 ms) rather than the whole recording. Sequential playback pays
that once per block.

The bottom two bits are dropped before compressing, which is lossless for this
sensor: the P3 only ever emits multiples of 4, so its real quantum is 0.0625 C
rather than the 1/64 C the format implies.

`--no-compress` writes a plain `.raw` instead. Both formats open in replay and via
`thermal_store.open_frames`, so existing recordings keep working:

```python
import json, thermal_store
meta = json.load(open("run01/thermal.json"))
frames = thermal_store.open_frames("run01/thermal.tz")
celsius = frames[42] / 64.0 - 273.15      # frame 42, as degrees C
```

`combined.mp4` is written frame-for-frame from the thermal capture loop, taking
whatever the webcam last delivered. Both feeds therefore share one clock and the
file has a single consistent rate for the sound to align against.

Every recording is finalized through ffmpeg, which re-encodes to H.264 and corrects
the timing. These are working recordings rather than masters, so the default quality
(CRF 23) trades some compression artefacts for roughly a third of the size -- a
measured session went from 2.44 MB to 1.08 MB. The 16-bit `thermal.raw` is never
touched, so measurements stay exact.

`p3-viewer --record BASE` writes the same thermal trio without a folder, as
`BASE.mp4` / `BASE.raw` / `BASE.json`, and `R` toggles recording at runtime.

```bash
p3-viewer --range 20 350 --gain low --record run01
```

The raw stream is unfiltered so measurements stay faithful; temporal noise reduction is
visible in the mp4 only. Frame *i* of the mp4 corresponds to frame *i* of the raw file.
Read it back with the shape from the sidecar:

```python
import json, thermal_store
meta = json.load(open("run01/thermal.json"))
frames = thermal_store.open_frames(
    "run01/" + meta["raw_file"], meta["raw_shape"]
)
celsius = frames[0] / 64.0 - 273.15
```

#### Timestamp

Both videos carry a burned-in `YYYY-MM-DD HH:MM:SS` stamp in the top-right corner,
drawn from each frame's own capture time. The thermal and visible streams therefore
show the same wall-clock time on frames captured together, which is what makes them
correlatable despite running at different rates.

Turn it off with `--no-timestamp`, the **Date/time stamp** checkbox, or `T` in the
OpenCV viewer. It is burned into the rendered mp4 and the preview only -- the `.raw`
thermal stream is never touched, so measurements are unaffected.

`--record-fps` (default 25) only sets the mp4 container rate. Actual capture rate is
recorded as `measured_fps` in the sidecar; use it if playback speed matters.

**Controls:**

- `q` - Quit
- `h` - Toggle help overlay
- `c` - Cycle colormap
- `a` - Cycle AGC mode (factory -> percentile -> fixed range -> log range)
- `d` - Toggle DDE (detail enhancement)
- `p` - Toggle enhanced mode (CLAHE + DDE)
- `x` - Cycle scale/interpolation mode
- `t` - Toggle reticule
- `s` - Trigger shutter/NUC
- `g` - Toggle gain mode (high/low)
- `R` - Start/stop recording (mp4 + raw)
- `T` - Toggle the date/time stamp
- `r` - Rotate display 90°
- `m` - Mirror display
- `+`/`-` - Zoom in/out
- `e` - Cycle emissivity presets
- `1-9` - Set emissivity (0.1-0.9)
- `D` - Dump raw thermal data to file
- `Space` - Screenshot
- `l` - Activate lock-in thermography (see lock-in.md)
- `b` - Toggle min/max spot marker
- `v` - Toggle colorbar

### Library

```python
from p3_camera import Model, P3Camera, get_model_config, raw_to_celsius

# Use P3 camera (default)
camera = P3Camera()
# Or use P1 camera
# camera = P3Camera(config=get_model_config(Model.P1))

camera.connect()
camera.init()
camera.start_streaming()

ir_brightness, thermal_raw = camera.read_frame_both()
temps_celsius = raw_to_celsius(thermal_raw)

# Center coordinates depend on model
# P1: (59, 80), P3: (95, 128)
print(f"Center temp: {temps_celsius[temps_celsius.shape[0]//2, temps_celsius.shape[1]//2]:.1f}C")

camera.stop_streaming()
camera.disconnect()
```

## Protocol Documentation

See [P3_PROTOCOL.md](P3_PROTOCOL.md) for USB protocol details.

## Contributing

This project provides initial scaffolding for a P3 thermal camera application.
There's significant potential to build something great here, and contributions
are welcome!

Some areas that could use help:

- **macOS support** - USB handling on macOS
- **GUI application** - Qt/GTK interface beyond the OpenCV viewer
- **Recording/playback** - Video capture with thermal data preservation
- **Radiometric analysis** - Region statistics, spot meters, isotherms
- **Calibration tools** - Blackbody calibration, emissivity tables
- **Documentation** - Protocol details, hardware information

If you have a P3 camera and want to help improve this tool, PRs are welcome!

## License

Apache 2.0
