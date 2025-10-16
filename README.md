# Face Mask Detection GUI

A PyQt5-based desktop application that streams frames from your laptop camera, detects faces in real time using MediaPipe Face Mesh, segments full human bodies with MediaPipe Selfie Segmentation, and renders configurable masks over every detected region.

## Features

- Real-time face and full-body masking powered by MediaPipe Face Mesh and Selfie Segmentation, with configurable temporal and spatial smoothing for stable results
- Built-in GUI control panel to fine-tune accuracy parameters (confidence, smoothing, kernel size, model choice) while the stream is running
- Quick import/export, reset, and automatic persistence of your preferred accuracy settings between sessions
- Configurable camera index, refresh rate, mask opacity, mask color, and face-count/confidence thresholds
- Simple, responsive PyQt5 GUI with status messages
- Packaged with [Nix flakes](https://nixos.wiki/wiki/Flakes) for reproducible builds and development shells

## Prerequisites

- A working webcam accessible from the host operating system
- [Nix](https://nixos.org/download.html) with flakes enabled (`nix --experimental-features 'nix-command flakes'`)

## Quick start

Run the application directly via the flake:

```fish
nix run
```

You can pass command-line options through `--`, for example to use camera index `1`, a blue mask, and allow up to `3` faces:

```fish
nix run -- --camera 1 --mask-color 255,0,0 --max-faces 3
```

## Development shell

Enter a shell with Python, OpenCV, and PyQt5 available:

```fish
nix develop
```

Once inside, you can run the application manually:

```fish
python -m face_gui.app
```

## Command-line options

| Option | Description |
| --- | --- |
| `--camera` | Camera device index (default: `0`). |
| `--fps` | Target refresh rate in frames per second (default: `30`). |
| `--mask-alpha` | Opacity of the face mask overlay between `0` (transparent) and `1` (opaque). Default `1` for a solid fill. |
| `--mask-color` | Face mask color as comma-separated `B,G,R` values. Default `0,255,0` (green). |
| `--max-faces` | Maximum number of faces to detect simultaneously. Default `5`. |
| `--min-detection-confidence` | Minimum detection confidence threshold passed to MediaPipe (0–1). Default `0.5`. |
| `--min-tracking-confidence` | Minimum tracking confidence threshold for temporal smoothing (0–1). Default `0.5`. |
| `--disable-face-mask` | Skip drawing the face polygon mask (useful for body-only masking). |
| `--disable-body-mask` | Skip drawing the body segmentation mask. |
| `--segmentation-threshold` | Confidence threshold (0–1) for accepting body pixels. Default `0.5`. |
| `--segmentation-model` | MediaPipe Selfie Segmentation model index (`0` or `1`). Default `1`. |
| `--segmentation-smooth-factor` | Temporal smoothing factor (0–0.99) for segment mask averaging. Higher values reduce flicker but respond slower. Default `0.6`. |
| `--segmentation-kernel-size` | Odd kernel size for Gaussian blur and morphological cleanup. Larger values fill gaps but may bleed edges. Default `7`. |

### Improving accuracy

- Use the settings panel on the right side of the window to tweak the same parameters live without restarting the app.
- For close-up users, try `--segmentation-model 0`; for wider scenes, keep the (landscape) default `1`.
- Increase `--segmentation-smooth-factor` (e.g. `0.8`) to suppress flicker when subjects move slowly. Lower it for faster response.
- Adjust `--segmentation-kernel-size` to fill gaps (larger) or recover detail (smaller). Values must be odd; `7`–`11` works well for upper-body crops.
- Raise `--segmentation-threshold` (e.g. `0.6`) if you see background bleeding through, or lower it if limbs disappear in low light.

### Settings management

- The GUI slider panel mirrors the command-line options and automatically saves your latest choices under `~/.face_gui/settings.json` (or the OS-specific config directory) when you change them.
- Use the **Import…** and **Export…** buttons to store or reload JSON snapshots of these settings across machines.
- Hit **Reset** whenever you want to return to the default detection and segmentation parameters.

## Project layout

```
./
├── flake.nix          # Nix flake definition for builds and dev shells
├── pyproject.toml     # Python project metadata and dependencies
├── README.md          # This documentation
└── src/face_gui/
    ├── __init__.py
    └── app.py         # PyQt application entry point
```

## Troubleshooting

- **No camera feed:** Ensure no other program is using the webcam. On Linux you may need to grant permissions (e.g. `v4l2`).
- **Missing Qt platform plugin (`xcb`):** When running outside the flake, install the appropriate Qt platform packages for your distribution.
- **Poor detection quality:** Ensure your scene is well lit. Adjust `--mask-alpha` or consider fine-tuning detection parameters in `_annotate_faces`.

## License

MIT License (add your preferred license here).
