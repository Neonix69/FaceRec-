import os
import cv2
import json
import time
import queue
import shutil
import threading
import numpy as np
import customtkinter as ctk

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from datetime import datetime

from insightface.app import FaceAnalysis

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

KNOWN_FACES_DIR = "known_faces"
RESULTS_DIR = "results"
CAMERA_CAPTURES_DIR = "camera_captures"

CONFIG_PATH = "config.json"
HISTORY_PATH = "recognition_history.json"

MIN_ZOOM = 0.2
MAX_ZOOM = 6.0
ZOOM_STEP = 1.1

FACE_CROP_MARGIN = 0.3
DUPLICATE_WARNING_THRESHOLD = 0.5

MATCH_THRESHOLD = 0.45


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CAMERA_CAPTURES_DIR, exist_ok=True)


# ============================================================
# CONFIG / HISTORY PERSISTENCE
# ============================================================

def load_config():

    global MATCH_THRESHOLD

    if os.path.exists(CONFIG_PATH):

        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            MATCH_THRESHOLD = float(
                data.get("match_threshold", MATCH_THRESHOLD)
            )

        except Exception:
            pass


def save_config():

    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"match_threshold": MATCH_THRESHOLD}, f)

    except Exception as e:
        print(f"Could not save config: {e}")


history_log = []


def load_history():

    global history_log

    if os.path.exists(HISTORY_PATH):

        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history_log = json.load(f)

        except Exception:
            history_log = []


def save_history():

    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history_log, f, indent=2)

    except Exception as e:
        print(f"Could not save history: {e}")


def log_history(image_name, results):

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": image_name,
        "faces": [
            {"name": n, "score": round(float(s), 1)}
            for n, s, b in results
        ]
    }

    history_log.append(entry)

    del history_log[:-200]

    save_history()


load_config()
load_history()


# ============================================================
# GLOBAL STATE
# ============================================================

app = None

known_faces = {}

current_image = None
current_result = None
result_saved = False

current_face_results = []

full_res_image = None
zoom_scale = 1.0
base_fit_scale = 1.0
tk_photo = None
canvas_image_id = None

drag_start = (0, 0)
drag_moved = False

hover_face_index = None
hover_rect_id = None
hover_tooltip_id = None

last_added = {"path": None, "backup": None}

model_load_queue = queue.Queue()
batch_queue = queue.Queue()
camera_queue = queue.Queue(maxsize=2)

camera_active = False
camera_capture = None
camera_thread = None
camera_stop_flag = threading.Event()
latest_camera_frame = {"annotated": None, "results": None, "raw": None}

busy = True

MAIN_ACTION_BUTTONS = []


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================

def normalize_embedding(embedding):

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def cosine_similarity(embedding1, embedding2):

    return np.dot(embedding1, embedding2)


def sanitize_name(raw):

    return "".join(
        c for c in raw.strip()
        if c.isalnum() or c in (" ", "_", "-")
    ).strip()


def find_similar_known(embedding, exclude_name=None, threshold=DUPLICATE_WARNING_THRESHOLD):

    best_name, best_sim = None, -1

    for name, known_embedding in known_faces.items():

        if name == exclude_name:
            continue

        sim = cosine_similarity(embedding, known_embedding)

        if sim > best_sim:
            best_sim = sim
            best_name = name

    if best_name is not None and best_sim >= threshold:
        return best_name, best_sim

    return None, best_sim


def find_known_face_file(name):

    for filename in os.listdir(KNOWN_FACES_DIR):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        if os.path.splitext(filename)[0] == name:
            return os.path.join(KNOWN_FACES_DIR, filename)

    return None


# ============================================================
# LOAD KNOWN FACES
# ============================================================

def load_known_faces():

    global known_faces

    known_faces.clear()

    if app is None:
        return

    for filename in os.listdir(KNOWN_FACES_DIR):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        path = os.path.join(KNOWN_FACES_DIR, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        faces = app.get(image)

        if len(faces) == 0:
            continue

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        embedding = normalize_embedding(face.embedding)

        name = os.path.splitext(filename)[0]

        known_faces[name] = embedding


# ============================================================
# RECOGNIZE FACE
# ============================================================

def recognize_face(face):

    if not known_faces:
        return "Unknown", 0.0

    current_embedding = normalize_embedding(face.embedding)

    best_name = "Unknown"
    best_similarity = -1

    for name, known_embedding in known_faces.items():

        similarity = cosine_similarity(current_embedding, known_embedding)

        if similarity > best_similarity:
            best_similarity = similarity
            best_name = name

    if best_similarity >= MATCH_THRESHOLD:
        return best_name, best_similarity

    return "Unknown", best_similarity


# ============================================================
# DRAW BOXES / PROCESS IMAGE
# ============================================================

def draw_face_boxes(image, faces):

    results = []

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        name, similarity = recognize_face(face)

        score = max(0, similarity) * 100

        results.append((name, score, (x1, y1, x2, y2)))

        box_color = (0, 165, 255) if name == "Unknown" else (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)

        label = f"{name}  {score:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y1 - 10, text_height + 10)

        cv2.rectangle(
            image,
            (x1, label_y - text_height - 10),
            (x1 + text_width + 10, label_y + 5),
            box_color,
            -1
        )

        cv2.putText(
            image, label, (x1 + 5, label_y),
            font, font_scale, (0, 0, 0), thickness
        )

    return image, results


def recognize_image_core(image_path):
    """Thread-safe: no Tk calls. Returns (result_image_or_None, results_list)."""

    image = cv2.imread(image_path)

    if image is None or app is None:
        return None, []

    faces = app.get(image)

    return draw_face_boxes(image, faces)


def recognize_image(image_path):
    """Main-thread wrapper: shows dialogs on failure."""

    if app is None:
        messagebox.showwarning("Model Loading", "The AI model is still loading. Please wait.")
        return None, []

    result_image, results = recognize_image_core(image_path)

    if result_image is None:
        messagebox.showerror("Error", "Could not open the selected image.")

    return result_image, results


# ============================================================
# CANVAS RENDERING (zoom + pan)
# ============================================================

def clear_hover():

    global hover_rect_id, hover_tooltip_id, hover_face_index

    if hover_rect_id is not None:
        canvas.delete(hover_rect_id)
        hover_rect_id = None

    if hover_tooltip_id is not None:
        canvas.delete(hover_tooltip_id)
        hover_tooltip_id = None

    hover_face_index = None

    canvas.configure(cursor="")


def redraw_canvas():

    global tk_photo, canvas_image_id

    clear_hover()

    if full_res_image is None:
        return

    total_scale = base_fit_scale * zoom_scale

    new_w = max(1, int(full_res_image.width * total_scale))
    new_h = max(1, int(full_res_image.height * total_scale))

    resized = full_res_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    tk_photo = ImageTk.PhotoImage(resized)

    canvas.delete("img")

    canvas_image_id = canvas.create_image(0, 0, image=tk_photo, anchor="nw", tags="img")

    canvas.configure(scrollregion=(0, 0, new_w, new_h))


def show_placeholder():

    global full_res_image, tk_photo, canvas_image_id

    clear_hover()
    canvas.delete("all")

    full_res_image = None
    tk_photo = None
    canvas_image_id = None

    canvas.update_idletasks()

    w = canvas.winfo_width() or 850
    h = canvas.winfo_height() or 500

    msg = "No image selected\n\nClick 'Select Image' to begin"

    if DND_AVAILABLE:
        msg += "\nor drag & drop an image here"

    canvas.create_text(
        w / 2, h / 2, text=msg, fill="gray70",
        font=("Segoe UI", 16), justify="center", tags="placeholder"
    )

    canvas.configure(scrollregion=(0, 0, w, h))


def load_new_image_into_canvas(cv_image):

    global full_res_image, zoom_scale, base_fit_scale

    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    full_res_image = Image.fromarray(rgb)

    zoom_scale = 1.0

    canvas.update_idletasks()

    canvas_w = canvas.winfo_width() or 850
    canvas_h = canvas.winfo_height() or 500

    base_fit_scale = min(
        canvas_w / full_res_image.width,
        canvas_h / full_res_image.height,
        1.0
    )

    if base_fit_scale <= 0:
        base_fit_scale = 1.0

    redraw_canvas()

    canvas.xview_moveto(0)
    canvas.yview_moveto(0)


def refresh_canvas_image(cv_image):

    global full_res_image

    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    full_res_image = Image.fromarray(rgb)

    redraw_canvas()


def reset_zoom():

    global zoom_scale

    if full_res_image is None:
        return

    zoom_scale = 1.0

    redraw_canvas()

    canvas.xview_moveto(0)
    canvas.yview_moveto(0)


# ============================================================
# ZOOM (mouse wheel + trackpad)
# ============================================================

def _apply_zoom(delta_positive, anchor_x, anchor_y):

    global zoom_scale

    if full_res_image is None:
        return

    old_scale = zoom_scale
    old_total = base_fit_scale * old_scale

    if delta_positive:
        zoom_scale = min(zoom_scale * ZOOM_STEP, MAX_ZOOM)
    else:
        zoom_scale = max(zoom_scale / ZOOM_STEP, MIN_ZOOM)

    if zoom_scale == old_scale:
        return

    cx = canvas.canvasx(anchor_x)
    cy = canvas.canvasy(anchor_y)

    img_x = cx / old_total if old_total else 0
    img_y = cy / old_total if old_total else 0

    redraw_canvas()

    new_total = base_fit_scale * zoom_scale

    total_w = full_res_image.width * new_total
    total_h = full_res_image.height * new_total

    if total_w > 0:
        new_cx = img_x * new_total
        canvas.xview_moveto(max(0, min(1, (new_cx - anchor_x) / total_w)))

    if total_h > 0:
        new_cy = img_y * new_total
        canvas.yview_moveto(max(0, min(1, (new_cy - anchor_y) / total_h)))


def on_mousewheel(event):
    _apply_zoom(event.delta > 0, event.x, event.y)


def on_mousewheel_linux_up(event):
    _apply_zoom(True, event.x, event.y)


def on_mousewheel_linux_down(event):
    _apply_zoom(False, event.x, event.y)


# ============================================================
# PAN (drag) + click-to-name detection
# ============================================================

def on_canvas_press(event):

    global drag_start, drag_moved

    drag_start = (event.x, event.y)
    drag_moved = False

    canvas.scan_mark(event.x, event.y)


def on_canvas_drag(event):

    global drag_moved

    if abs(event.x - drag_start[0]) > 4 or abs(event.y - drag_start[1]) > 4:
        drag_moved = True

    canvas.scan_dragto(event.x, event.y, gain=1)

    on_canvas_motion(event)


def on_canvas_release(event):

    if not drag_moved and hover_face_index is not None:
        prompt_name_face(hover_face_index)


def on_canvas_motion(event):

    global hover_face_index, hover_rect_id, hover_tooltip_id

    if camera_active:
        return

    if not current_face_results or full_res_image is None:
        if hover_face_index is not None:
            clear_hover()
        return

    total_scale = base_fit_scale * zoom_scale

    cx = canvas.canvasx(event.x)
    cy = canvas.canvasy(event.y)

    img_x = cx / total_scale if total_scale else 0
    img_y = cy / total_scale if total_scale else 0

    found_index = None

    for i, (name, score, bbox) in enumerate(current_face_results):

        if name != "Unknown":
            continue

        x1, y1, x2, y2 = bbox

        if x1 <= img_x <= x2 and y1 <= img_y <= y2:
            found_index = i
            break

    if found_index == hover_face_index:

        if found_index is not None and hover_tooltip_id is not None:
            canvas.coords(hover_tooltip_id, cx + 12, cy - 12)

        return

    clear_hover()

    hover_face_index = found_index

    if found_index is not None:

        name, score, bbox = current_face_results[found_index]
        x1, y1, x2, y2 = bbox

        hover_rect_id = canvas.create_rectangle(
            x1 * total_scale, y1 * total_scale,
            x2 * total_scale, y2 * total_scale,
            outline="yellow", width=3
        )

        hover_tooltip_id = canvas.create_text(
            cx + 12, cy - 12, text="Click to name this face",
            fill="yellow", anchor="w", font=("Segoe UI", 11, "bold")
        )

        canvas.configure(cursor="hand2")


def on_canvas_configure(event):

    if full_res_image is None:
        show_placeholder()


def on_file_drop(event):

    if busy:
        return

    try:
        paths = root.tk.splitlist(event.data)
    except Exception:
        paths = [event.data]

    if not paths:
        return

    path = paths[0].strip("{}")

    if not path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        messagebox.showwarning(
            "Unsupported File",
            "Please drop an image file (jpg, jpeg, png, webp)."
        )
        return

    select_image(path=path)


# ============================================================
# NAME AN UNKNOWN FACE -> ADD TO known_faces
# ============================================================

def prompt_name_face(index):

    if app is None or index is None or index >= len(current_face_results):
        return

    name, score, bbox = current_face_results[index]

    if name != "Unknown":
        return

    new_name = simpledialog.askstring(
        "Name This Face", "Enter a name for this person:", parent=root
    )

    if not new_name:
        return

    safe_name = sanitize_name(new_name)

    if not safe_name:
        messagebox.showerror("Invalid Name", "Please enter a valid name.")
        return

    source_image = cv2.imread(current_image)

    if source_image is None:
        messagebox.showerror("Error", "Could not reload the original image to crop the face.")
        return

    h, w = source_image.shape[:2]
    x1, y1, x2, y2 = bbox

    margin_x = int((x2 - x1) * FACE_CROP_MARGIN)
    margin_y = int((y2 - y1) * FACE_CROP_MARGIN)

    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_x2 = min(w, x2 + margin_x)
    crop_y2 = min(h, y2 + margin_y)

    face_crop = source_image[crop_y1:crop_y2, crop_x1:crop_x2]

    if face_crop.size == 0:
        messagebox.showerror("Error", "Could not crop the face from the image.")
        return

    crop_faces = app.get(face_crop)

    if crop_faces:

        crop_embedding = normalize_embedding(
            max(crop_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).embedding
        )

        dup_name, dup_sim = find_similar_known(crop_embedding, exclude_name=safe_name)

        if dup_name is not None:

            proceed = messagebox.askyesno(
                "Possible Duplicate",
                f"This face looks similar to already-known '{dup_name}' "
                f"({dup_sim * 100:.0f}% similar).\n\n"
                f"Save as a new person '{safe_name}' anyway?"
            )

            if not proceed:
                return

    dest_path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}.jpg")
    backup = None

    if os.path.exists(dest_path):

        if not messagebox.askyesno(
            "Face Already Exists",
            f"A known face named '{safe_name}' already exists.\n\nOverwrite it?"
        ):
            return

        with open(dest_path, "rb") as f:
            backup = f.read()

    cv2.imwrite(dest_path, face_crop)

    last_added["path"] = dest_path
    last_added["backup"] = backup
    undo_button.configure(state="normal")

    load_known_faces()

    status_label.configure(
        text=f"Added '{safe_name}' to known faces • {len(known_faces)} known"
    )

    run_recognition()


def undo_last_add():

    if last_added["path"] is None:
        return

    path = last_added["path"]
    backup = last_added["backup"]

    if backup is not None:
        with open(path, "wb") as f:
            f.write(backup)
    else:
        if os.path.exists(path):
            os.remove(path)

    last_added["path"] = None
    last_added["backup"] = None

    undo_button.configure(state="disabled")

    load_known_faces()

    status_label.configure(text="Undid last name/add ✓")

    if current_image is not None:
        run_recognition()


# ============================================================
# MANAGE KNOWN FACES (in-app panel)
# ============================================================

def load_face_thumbnail(name):

    path = find_known_face_file(name)

    if path is None:
        return None

    img = cv2.imread(path)

    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((48, 48), Image.Resampling.LANCZOS)

    return ImageTk.PhotoImage(pil)


def rename_known_face(name, refresh_cb):

    old_path = find_known_face_file(name)

    if old_path is None:
        messagebox.showerror("Error", f"Could not find the file for '{name}'.")
        return

    new_name = simpledialog.askstring(
        "Rename Face", f"New name for '{name}':", initialvalue=name, parent=root
    )

    if not new_name:
        return

    safe_name = sanitize_name(new_name)

    if not safe_name or safe_name == name:
        return

    ext = os.path.splitext(old_path)[1]
    new_path = os.path.join(KNOWN_FACES_DIR, safe_name + ext)

    if os.path.exists(new_path):

        if not messagebox.askyesno(
            "Face Already Exists", f"'{safe_name}' already exists. Overwrite it?"
        ):
            return

        os.remove(new_path)

    os.rename(old_path, new_path)

    load_known_faces()
    refresh_cb()

    status_label.configure(text=f"Renamed '{name}' to '{safe_name}' ✓")


def delete_known_face(name, refresh_cb):

    if not messagebox.askyesno(
        "Delete Known Face", f"Delete '{name}' from known faces? This can't be undone."
    ):
        return

    path = find_known_face_file(name)

    if path and os.path.exists(path):
        os.remove(path)

    load_known_faces()
    refresh_cb()

    status_label.configure(text=f"Deleted '{name}' • {len(known_faces)} known")


def add_known_face_from_file(refresh_cb):

    if app is None:
        messagebox.showwarning("Model Loading", "Please wait for the AI model to finish loading.")
        return

    file_path = filedialog.askopenfilename(
        title="Select a Face Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")]
    )

    if not file_path:
        return

    image = cv2.imread(file_path)

    if image is None:
        messagebox.showerror("Error", "Could not read the selected image.")
        return

    faces = app.get(image)

    if not faces:
        messagebox.showerror("No Face Found", "No face was detected in that image.")
        return

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = normalize_embedding(face.embedding)

    new_name = simpledialog.askstring(
        "Name This Face", "Enter a name for this person:", parent=root
    )

    if not new_name:
        return

    safe_name = sanitize_name(new_name)

    if not safe_name:
        messagebox.showerror("Invalid Name", "Please enter a valid name.")
        return

    dup_name, dup_sim = find_similar_known(embedding, exclude_name=safe_name)

    if dup_name is not None:

        if not messagebox.askyesno(
            "Possible Duplicate",
            f"This face looks similar to already-known '{dup_name}' "
            f"({dup_sim * 100:.0f}% similar).\n\nSave as a new person '{safe_name}' anyway?"
        ):
            return

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"

    dest_path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}{ext}")
    backup = None

    if os.path.exists(dest_path):

        if not messagebox.askyesno(
            "Face Already Exists", f"'{safe_name}' already exists. Overwrite it?"
        ):
            return

        with open(dest_path, "rb") as f:
            backup = f.read()

    shutil.copyfile(file_path, dest_path)

    last_added["path"] = dest_path
    last_added["backup"] = backup
    undo_button.configure(state="normal")

    load_known_faces()
    refresh_cb()

    status_label.configure(text=f"Added '{safe_name}' to known faces • {len(known_faces)} known")


def open_manage_known_faces():

    if app is None:
        messagebox.showwarning("Model Loading", "Please wait for the AI model to finish loading.")
        return

    win = ctk.CTkToplevel(root)
    win.title("Manage Known Faces")
    win.geometry("460x560")
    win.transient(root)

    header_lbl = ctk.CTkLabel(win, text="Known Faces", font=ctk.CTkFont(size=18, weight="bold"))
    header_lbl.pack(pady=(15, 5))

    scroll_frame = ctk.CTkScrollableFrame(win, width=420, height=440)

    thumb_refs = []

    def rebuild():

        for widget in scroll_frame.winfo_children():
            widget.destroy()

        thumb_refs.clear()

        if not known_faces:
            ctk.CTkLabel(scroll_frame, text="No known faces yet.").pack(pady=20)
            return

        for name in sorted(known_faces.keys()):

            row = ctk.CTkFrame(scroll_frame)
            row.pack(fill="x", pady=4)

            thumb_img = load_face_thumbnail(name)

            if thumb_img is not None:
                thumb_refs.append(thumb_img)
                ctk.CTkLabel(row, image=thumb_img, text="").pack(side="left", padx=8, pady=6)
            else:
                ctk.CTkLabel(row, text="?", width=48).pack(side="left", padx=8, pady=6)

            ctk.CTkLabel(row, text=name, anchor="w").pack(side="left", fill="x", expand=True, padx=8)

            ctk.CTkButton(
                row, text="Rename", width=70,
                command=lambda n=name: rename_known_face(n, rebuild)
            ).pack(side="left", padx=4)

            ctk.CTkButton(
                row, text="Delete", width=70,
                fg_color="#8b3a3a", hover_color="#722f2f",
                command=lambda n=name: delete_known_face(n, rebuild)
            ).pack(side="left", padx=(4, 8))

    add_btn = ctk.CTkButton(
        win, text="Add From File...", command=lambda: add_known_face_from_file(rebuild)
    )
    add_btn.pack(pady=(0, 10), padx=20, fill="x")

    scroll_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    rebuild()


# ============================================================
# HISTORY VIEWER (in-app panel)
# ============================================================

def open_history():

    win = ctk.CTkToplevel(root)
    win.title("Recognition History")
    win.geometry("480x560")
    win.transient(root)

    ctk.CTkLabel(win, text="Recent Recognitions", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15, 10))

    scroll = ctk.CTkScrollableFrame(win, width=440, height=460)
    scroll.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    if not history_log:

        ctk.CTkLabel(scroll, text="No recognitions logged yet.").pack(pady=20)

    else:

        for entry in reversed(history_log[-100:]):

            if entry["faces"]:
                names = ", ".join(f"{f['name']} ({f['score']:.0f}%)" for f in entry["faces"])
            else:
                names = "No faces"

            row_text = f"{entry['timestamp']}\n{entry['image']}\n{names}"

            ctk.CTkLabel(
                scroll, text=row_text, justify="left", anchor="w",
                font=ctk.CTkFont(size=12)
            ).pack(fill="x", pady=6, padx=6)

            ctk.CTkFrame(scroll, height=1, fg_color="gray30").pack(fill="x", padx=6)


# ============================================================
# SELECT IMAGE
# ============================================================

def select_image(path=None):

    global current_image, current_result, result_saved

    if busy:
        messagebox.showinfo("Please Wait", "Please wait for the current operation to finish.")
        return

    if current_result is not None and not result_saved:

        choice = messagebox.askyesnocancel(
            "Unsaved Result",
            "The current recognition result has not been saved.\n\n"
            "Do you want to save it before selecting another image?"
        )

        if choice is None:
            return

        if choice:
            if not save_result():
                return

    if path is None:

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")]
        )

        if not file_path:
            return

    else:

        file_path = path

        if not os.path.exists(file_path):
            messagebox.showerror("Error", "That file could not be found.")
            return

    current_image = file_path
    current_result = None
    current_face_results.clear()
    result_saved = False

    original = cv2.imread(file_path)

    if original is None:
        messagebox.showerror("Error", "Could not read that image file.")
        return

    load_new_image_into_canvas(original)

    results_text.configure(text="")
    status_label.configure(text="Image selected. Click Recognize Faces.")


# ============================================================
# RUN RECOGNITION
# ============================================================

def render_recognition_results(results):

    if not results:
        text = "No faces detected."
    else:
        text = f"Faces detected: {len(results)}\n\n"

        for i, (name, score, bbox) in enumerate(results, start=1):
            text += f"{i}. {name} ({score:.1f}%)\n"

        if any(n == "Unknown" for n, s, b in results):
            text += "\nHover an orange face, then click to name it."

    results_text.configure(text=text)
    status_label.configure(text="Recognition complete • Unsaved")


def run_recognition():

    global current_result, result_saved

    if busy:
        messagebox.showinfo("Please Wait", "Please wait for the current operation to finish.")
        return

    if current_image is None:
        messagebox.showwarning("No Image", "Please select an image first.")
        return

    status_label.configure(text="Recognizing faces...")
    root.update()

    result_image, results = recognize_image(current_image)

    if result_image is None:
        return

    current_result = result_image
    current_face_results.clear()
    current_face_results.extend(results)
    result_saved = False

    refresh_canvas_image(result_image)
    render_recognition_results(results)
    log_history(os.path.basename(current_image), results)


# ============================================================
# SAVE RESULT
# ============================================================

def save_result():

    global result_saved

    if current_result is None:
        messagebox.showwarning("Nothing to Save", "Run face recognition first.")
        return False

    filename = filedialog.asksaveasfilename(
        title="Save Result",
        initialdir=RESULTS_DIR,
        defaultextension=".jpg",
        filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")]
    )

    if not filename:
        return False

    if not cv2.imwrite(filename, current_result):
        messagebox.showerror("Save Error", "Could not save the result.")
        return False

    result_saved = True

    status_label.configure(text="Result saved ✓")
    messagebox.showinfo("Saved", f"Result saved successfully:\n\n{filename}")

    return True


# ============================================================
# CLEAR / NEW IMAGE
# ============================================================

def clear_current_image():

    global current_image, current_result, result_saved

    current_image = None
    current_result = None
    current_face_results.clear()
    result_saved = False

    show_placeholder()

    results_text.configure(text="")
    status_label.configure(text=f"Ready • {len(known_faces)} known face(s)")


def new_image():

    if current_result is None or result_saved:
        clear_current_image()
        return

    choice = messagebox.askyesnocancel(
        "Unsaved Result",
        "The current recognition result has not been saved.\n\n"
        "Do you want to save it before starting a new one?"
    )

    if choice is None:
        return

    if choice:
        if not save_result():
            return

    clear_current_image()


def refresh_faces():

    load_known_faces()

    status_label.configure(text=f"Loaded {len(known_faces)} known face(s) ✓")


# ============================================================
# CONTROLS ENABLE/DISABLE
# ============================================================

def set_controls_enabled(enabled):

    state = "normal" if enabled else "disabled"

    for widget in MAIN_ACTION_BUTTONS:
        try:
            widget.configure(state=state)
        except Exception:
            pass


# ============================================================
# PROGRESS BAR HELPERS
# ============================================================

def set_progress_indeterminate(text):

    progress_label.configure(text=text)
    progress_bar.configure(mode="indeterminate")
    progress_bar.start()


def set_progress_determinate(fraction, text):

    progress_bar.stop()
    progress_bar.configure(mode="determinate")
    progress_bar.set(fraction)
    progress_label.configure(text=text)


def clear_progress():

    progress_bar.stop()
    progress_bar.configure(mode="determinate")
    progress_bar.set(0)
    progress_label.configure(text="")


# ============================================================
# THRESHOLD SLIDER
# ============================================================

def on_threshold_change(value):

    global MATCH_THRESHOLD

    MATCH_THRESHOLD = round(float(value), 2)

    threshold_label.configure(text=f"Match Threshold: {MATCH_THRESHOLD:.2f}")

    save_config()


# ============================================================
# BATCH PROCESSING
# ============================================================

def batch_worker(image_files):

    total = len(image_files)
    faces_total = 0
    unknown_total = 0

    for i, path in enumerate(image_files, start=1):

        try:
            result_image, results = recognize_image_core(path)

            if result_image is not None:

                out_path = os.path.join(RESULTS_DIR, f"result_{os.path.basename(path)}")
                cv2.imwrite(out_path, result_image)

                faces_total += len(results)
                unknown_total += sum(1 for n, s, b in results if n == "Unknown")

                log_history(os.path.basename(path), results)

        except Exception as e:
            print(f"Batch error on {path}: {e}")

        batch_queue.put(("progress", i, total))

    batch_queue.put(("done", total, faces_total, unknown_total))


def poll_batch_queue():

    global busy

    try:

        while True:

            msg = batch_queue.get_nowait()

            if msg[0] == "progress":
                _, i, total = msg
                set_progress_determinate(i / total, f"Processing {i}/{total}...")

            elif msg[0] == "done":

                _, total, faces_total, unknown_total = msg

                clear_progress()
                set_controls_enabled(True)
                busy = False

                status_label.configure(text=f"Batch complete • {total} images processed")

                messagebox.showinfo(
                    "Batch Complete",
                    f"Processed {total} image(s).\n"
                    f"Faces detected: {faces_total}\n"
                    f"Unknown: {unknown_total}\n\n"
                    f"Results saved in '{RESULTS_DIR}'."
                )

                return

    except queue.Empty:
        pass

    root.after(150, poll_batch_queue)


def start_batch():

    global busy

    if busy:
        messagebox.showinfo("Please Wait", "Please wait for the current operation to finish.")
        return

    if app is None:
        messagebox.showwarning("Model Loading", "Please wait for the AI model to finish loading.")
        return

    folder = filedialog.askdirectory(title="Select a Folder of Images to Process")

    if not folder:
        return

    image_files = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    if not image_files:
        messagebox.showinfo("No Images", "No image files were found in that folder.")
        return

    busy = True
    set_controls_enabled(False)
    set_progress_determinate(0, f"Processing 0/{len(image_files)}...")

    threading.Thread(target=batch_worker, args=(image_files,), daemon=True).start()
    poll_batch_queue()


# ============================================================
# CAMERA (LIVE)
# ============================================================

def camera_loop():

    global camera_capture

    frame_count = 0

    try:

        while not camera_stop_flag.is_set():

            if camera_capture is None:
                break

            ret, frame = camera_capture.read()

            if not ret:
                break

            frame_count += 1

            if app is not None and frame_count % 2 == 0:
                annotated, results = draw_face_boxes(frame.copy(), app.get(frame))
            else:
                annotated, results = frame, []

            try:
                if camera_queue.full():
                    camera_queue.get_nowait()
                camera_queue.put_nowait((annotated, results, frame))
            except queue.Full:
                pass

            time.sleep(0.03)

    finally:

        if camera_capture is not None:
            camera_capture.release()
            camera_capture = None


def poll_camera_queue():

    if not camera_active:
        return

    try:
        while True:
            annotated, results, raw = camera_queue.get_nowait()
            latest_camera_frame["annotated"] = annotated
            latest_camera_frame["results"] = results
            latest_camera_frame["raw"] = raw
    except queue.Empty:
        pass

    if latest_camera_frame["annotated"] is not None:
        refresh_canvas_image(latest_camera_frame["annotated"])

    root.after(60, poll_camera_queue)


def start_camera():

    global camera_capture, camera_active, camera_thread, busy

    if busy:
        messagebox.showinfo("Please Wait", "Please wait for the current operation to finish.")
        return

    if app is None:
        messagebox.showwarning("Model Loading", "Please wait for the AI model to finish loading.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror(
            "Camera Error",
            "Could not open a webcam. Check that one is connected and not already in use."
        )
        return

    camera_capture = cap
    camera_active = True
    busy = True

    camera_stop_flag.clear()

    set_controls_enabled(False)
    camera_button.configure(state="normal", text="Stop Camera")
    capture_frame_button.configure(state="normal")

    status_label.configure(text="Live camera running... (may lag on CPU)")

    latest_camera_frame["annotated"] = None
    latest_camera_frame["results"] = None
    latest_camera_frame["raw"] = None

    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()

    poll_camera_queue()


def stop_camera():

    global camera_active, busy

    if not camera_active:
        return

    camera_active = False
    camera_stop_flag.set()

    if camera_thread is not None:
        camera_thread.join(timeout=1.5)

    camera_button.configure(text="Start Camera")
    capture_frame_button.configure(state="disabled")

    set_controls_enabled(True)
    busy = False

    status_label.configure(text=f"Ready • {len(known_faces)} known face(s)")


def toggle_camera():

    if camera_active:
        stop_camera()
    else:
        start_camera()


def capture_frame():

    global current_image, current_result, result_saved

    raw = latest_camera_frame.get("raw")

    if raw is None:
        messagebox.showwarning("No Frame", "No camera frame is available yet.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CAMERA_CAPTURES_DIR, f"capture_{timestamp}.jpg")

    cv2.imwrite(path, raw)

    stop_camera()

    current_image = path

    result_image, results = recognize_image(path)

    if result_image is None:
        return

    current_result = result_image
    current_face_results.clear()
    current_face_results.extend(results)
    result_saved = False

    load_new_image_into_canvas(cv2.imread(path))
    refresh_canvas_image(result_image)
    render_recognition_results(results)
    log_history(os.path.basename(path), results)


# ============================================================
# MODEL LOADING (background thread)
# ============================================================

def load_model_bg():

    global app

    try:

        instance = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        instance.prepare(ctx_id=0, det_size=(640, 640))

        app = instance

        model_load_queue.put(("ok",))

    except Exception as e:
        model_load_queue.put(("error", str(e)))


def poll_model_load():

    global busy

    try:
        msg = model_load_queue.get_nowait()
    except queue.Empty:
        root.after(150, poll_model_load)
        return

    if msg[0] == "ok":

        clear_progress()

        load_known_faces()

        set_controls_enabled(True)
        busy = False

        status_label.configure(text=f"Ready • {len(known_faces)} known face(s)")

    else:

        clear_progress()
        status_label.configure(text="Model failed to load")

        messagebox.showerror("Model Load Error", f"Could not load the AI model:\n\n{msg[1]}")


# ============================================================
# MISC
# ============================================================

def open_results_folder():

    try:
        os.startfile(os.path.abspath(RESULTS_DIR))
    except Exception:
        try:
            import subprocess
            import sys
            if sys.platform == "darwin":
                subprocess.Popen(["open", RESULTS_DIR])
            else:
                subprocess.Popen(["xdg-open", RESULTS_DIR])
        except Exception:
            messagebox.showinfo("Results Folder", f"Results are saved in:\n{os.path.abspath(RESULTS_DIR)}")


def on_close():

    camera_stop_flag.set()

    if camera_capture is not None:
        try:
            camera_capture.release()
        except Exception:
            pass

    save_config()
    save_history()

    root.destroy()


# ============================================================
# GUI CONFIGURATION
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# MAIN WINDOW
# ============================================================

if DND_AVAILABLE:

    try:

        class _DnDRoot(ctk.CTk, TkinterDnD.DnDWrapper):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.TkdndVersion = TkinterDnD._require(self)

        root = _DnDRoot()

    except Exception as e:
        print(f"Drag-and-drop unavailable: {e}")
        DND_AVAILABLE = False
        root = ctk.CTk()

else:
    root = ctk.CTk()

root.title("FaceRec")
root.geometry("1300x800")
root.minsize(1100, 700)

root.protocol("WM_DELETE_WINDOW", on_close)


# ============================================================
# HEADER
# ============================================================

header = ctk.CTkFrame(root, corner_radius=0)
header.pack(fill="x")

title = ctk.CTkLabel(header, text="FaceRec", font=ctk.CTkFont(size=28, weight="bold"))
title.pack(pady=(20, 5))

subtitle = ctk.CTkLabel(header, text="Facial Recognition System", font=ctk.CTkFont(size=14))
subtitle.pack(pady=(0, 20))


# ============================================================
# MAIN AREA
# ============================================================

main_frame = ctk.CTkFrame(root, fg_color="transparent")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)


# ============================================================
# IMAGE PANEL (zoom/pan canvas)
# ============================================================

image_frame = ctk.CTkFrame(main_frame)
image_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

canvas = tk.Canvas(image_frame, bg="#1a1a1a", highlightthickness=0)
canvas.pack(fill="both", expand=True, padx=10, pady=10)

canvas.bind("<MouseWheel>", on_mousewheel)
canvas.bind("<Control-MouseWheel>", on_mousewheel)
canvas.bind("<Button-4>", on_mousewheel_linux_up)
canvas.bind("<Button-5>", on_mousewheel_linux_down)

canvas.bind("<ButtonPress-1>", on_canvas_press)
canvas.bind("<B1-Motion>", on_canvas_drag)
canvas.bind("<ButtonRelease-1>", on_canvas_release)
canvas.bind("<Motion>", on_canvas_motion)
canvas.bind("<Configure>", on_canvas_configure)

if DND_AVAILABLE:

    try:
        canvas.drop_target_register(DND_FILES)
        canvas.dnd_bind("<<Drop>>", on_file_drop)
    except Exception as e:
        print(f"Could not enable drag-and-drop: {e}")
        DND_AVAILABLE = False


# ============================================================
# CONTROL PANEL
# ============================================================

control_frame = ctk.CTkFrame(main_frame, width=310)
control_frame.pack(side="right", fill="y", padx=(10, 0))
control_frame.pack_propagate(False)

panel_title = ctk.CTkLabel(control_frame, text="Controls", font=ctk.CTkFont(size=22, weight="bold"))
panel_title.pack(pady=(20, 10))

scroll_area = ctk.CTkScrollableFrame(control_frame, width=280)
scroll_area.pack(fill="both", expand=True, padx=5, pady=(0, 5))


# ---- Workflow ----

select_button = ctk.CTkButton(scroll_area, text="Select Image", height=42, command=select_image)
select_button.pack(fill="x", padx=20, pady=6)

recognize_button = ctk.CTkButton(scroll_area, text="Recognize Faces", height=42, command=run_recognition)
recognize_button.pack(fill="x", padx=20, pady=6)

save_button = ctk.CTkButton(scroll_area, text="Save Result", height=42, command=save_result)
save_button.pack(fill="x", padx=20, pady=6)

reset_zoom_button = ctk.CTkButton(
    scroll_area, text="Reset Zoom", height=32, fg_color="transparent", border_width=1, command=reset_zoom
)
reset_zoom_button.pack(fill="x", padx=20, pady=(2, 10))


# ---- Match threshold ----

threshold_label = ctk.CTkLabel(scroll_area, text=f"Match Threshold: {MATCH_THRESHOLD:.2f}", font=ctk.CTkFont(size=13))
threshold_label.pack(anchor="w", padx=20, pady=(2, 0))

threshold_slider = ctk.CTkSlider(scroll_area, from_=0.20, to=0.80, number_of_steps=60, command=on_threshold_change)
threshold_slider.set(MATCH_THRESHOLD)
threshold_slider.pack(fill="x", padx=20, pady=(4, 12))


# ---- Tools ----

manage_button = ctk.CTkButton(scroll_area, text="Manage Known Faces", height=36, command=open_manage_known_faces)
manage_button.pack(fill="x", padx=20, pady=4)

batch_button = ctk.CTkButton(scroll_area, text="Batch Process Folder", height=36, command=start_batch)
batch_button.pack(fill="x", padx=20, pady=4)

camera_button = ctk.CTkButton(scroll_area, text="Start Camera", height=36, command=toggle_camera)
camera_button.pack(fill="x", padx=20, pady=4)

capture_frame_button = ctk.CTkButton(
    scroll_area, text="Capture Frame", height=32, state="disabled", command=capture_frame
)
capture_frame_button.pack(fill="x", padx=20, pady=(0, 4))

history_button = ctk.CTkButton(scroll_area, text="View History", height=36, command=open_history)
history_button.pack(fill="x", padx=20, pady=4)

results_folder_button = ctk.CTkButton(scroll_area, text="Open Results Folder", height=32, command=open_results_folder)
results_folder_button.pack(fill="x", padx=20, pady=(4, 12))


# ---- Progress ----

progress_label = ctk.CTkLabel(scroll_area, text="", font=ctk.CTkFont(size=12))
progress_label.pack(fill="x", padx=20, pady=(0, 0))

progress_bar = ctk.CTkProgressBar(scroll_area, mode="determinate")
progress_bar.set(0)
progress_bar.pack(fill="x", padx=20, pady=(2, 12))


# ---- Session ----

ctk.CTkFrame(scroll_area, height=1, fg_color="gray30").pack(fill="x", padx=20, pady=(0, 8))
ctk.CTkLabel(scroll_area, text="Session", font=ctk.CTkFont(size=12), text_color="gray60").pack(anchor="w", padx=20)

new_button = ctk.CTkButton(
    scroll_area, text="New", height=34, fg_color="gray30", hover_color="gray25", command=new_image
)
new_button.pack(fill="x", padx=20, pady=(6, 4))

refresh_button = ctk.CTkButton(
    scroll_area, text="Refresh Known Faces", height=34, fg_color="gray30", hover_color="gray25", command=refresh_faces
)
refresh_button.pack(fill="x", padx=20, pady=4)

undo_button = ctk.CTkButton(
    scroll_area, text="Undo Last Add", height=34, fg_color="gray30", hover_color="gray25",
    state="disabled", command=undo_last_add
)
undo_button.pack(fill="x", padx=20, pady=(4, 12))


# ---- Results ----

results_title = ctk.CTkLabel(scroll_area, text="Recognition Results", font=ctk.CTkFont(size=17, weight="bold"))
results_title.pack(pady=(6, 8))

results_text = ctk.CTkLabel(
    scroll_area, text="", justify="left", anchor="w", wraplength=250, font=ctk.CTkFont(size=14)
)
results_text.pack(fill="x", padx=20)


# ---- Status (pinned, outside scroll area) ----

status_label = ctk.CTkLabel(control_frame, text="Loading AI model...", font=ctk.CTkFont(size=13))
status_label.pack(side="bottom", pady=15)


# ============================================================
# BUTTON GROUP FOR ENABLE/DISABLE
# ============================================================

MAIN_ACTION_BUTTONS = [
    select_button, recognize_button, save_button, reset_zoom_button,
    manage_button, batch_button, camera_button, history_button,
    results_folder_button, new_button, refresh_button
]

set_controls_enabled(False)


# ============================================================
# KEYBOARD SHORTCUTS
# ============================================================

root.bind("<Control-o>", lambda e: select_image())
root.bind("<Control-r>", lambda e: run_recognition())
root.bind("<Control-s>", lambda e: save_result())


# ============================================================
# INITIAL STATE
# ============================================================

show_placeholder()

set_progress_indeterminate("Loading AI model...")

threading.Thread(target=load_model_bg, daemon=True).start()
poll_model_load()


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()
