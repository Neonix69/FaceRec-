# FaceRec

A face recognition desktop app built on [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l` model).

- **`face_recognition_gui.py`** — the main app. Everything is managed inside the GUI: select/recognize/save, a Known Faces manager (view, rename, delete, add), a recognition History panel, an adjustable match-threshold slider, batch folder processing, and a live webcam mode.
- **`face_recognition_system.py`** — a lightweight CLI fallback. Batch-processes every image in `test_images/` against `known_faces/` and writes results to `results/`. Useful for scripting/automation; the GUI is the recommended way to use this day-to-day.

## Setup

Requires Python **3.11** (InsightFace/onnxruntime don't yet reliably support 3.12+/3.14 on all platforms).

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

First launch downloads the `buffalo_l` model automatically (cached under `~/.insightface`), so it's slower the first time — the GUI shows a loading progress bar while this happens rather than freezing.

Optional: `pip install tkinterdnd2` to enable dragging an image file straight onto the app window. Without it, the file picker still works fine.

## Running it

```bash
python face_recognition_gui.py
```

### Core workflow
**Select Image → Recognize Faces → Save Result.** Scroll to zoom (centered on your cursor), drag to pan, "Reset Zoom" to snap back.

### Naming an unknown face
Hover an orange "Unknown" box on a recognized image and click it. Enter a name — it's cropped out of the original image (with a little padding for a better reference photo), checked against your existing known faces for possible duplicates, and saved. Known faces reload immediately. "Undo Last Add" reverses the most recent one if you made a mistake.

### Manage Known Faces
Opens a panel listing everyone currently known, with thumbnails. Rename or delete anyone, or add someone directly from a photo file — no need to touch the `known_faces/` folder by hand.

### View History
A running log of every recognition run (single image or batch), with timestamp, filename, and who was detected. Persists across sessions (`recognition_history.json`).

### Match Threshold
Slider in the control panel (default `0.45`, lower = stricter matching). Adjust live if you're getting false positives/negatives — no code editing needed. Saved across sessions (`config.json`).

### Batch Process Folder
Point it at any folder of images; it runs recognition on all of them, saves annotated copies into `results/`, and logs each to History. Runs in the background with a progress bar — the UI stays responsive.

### Live Camera
Starts your webcam with live recognition boxes drawn in real time (expect a few FPS on CPU-only inference, not full video-rate — it's genuinely running the model on frames, not a gimmick). "Capture Frame" freezes the current frame as a normal image you can save or hover-to-name faces on, just like a selected file.

## Project structure

```
FaceRec/
├── face_recognition_gui.py
├── face_recognition_system.py
├── known_faces/                 # reference photos, filename = person's name (managed via the GUI)
├── test_images/                 # images to scan (CLI mode only)
├── results/                     # annotated output
├── camera_captures/             # frames captured from the live camera
├── config.json                  # saved match threshold (auto-created)
├── recognition_history.json     # saved history log (auto-created)
├── requirements.txt
└── .gitignore
```

## Notes

- `known_faces/`, `test_images/`, `results/`, and `camera_captures/` are gitignored by default since they usually contain personal photos. Remove those lines from `.gitignore` if you want to version-control a shared face set.
- `config.json` and `recognition_history.json` are also gitignored (local, per-machine state).
- Keyboard shortcuts: `Ctrl+O` select image, `Ctrl+R` recognize, `Ctrl+S` save.
