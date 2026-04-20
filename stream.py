"""
Real-Time Pokédex — Flask app
- Captures webcam frames in a background thread
- Classifies them with YOLO11-cls (best.pt)
- Serves the video as MJPEG and the current detection as JSON
"""

import cv2
import time
import threading
import requests
from collections import deque
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

# ---------- Config ----------
MODEL_PATH = "best.pt"
CAMERA_INDEX = 0            # change to 1 or 2 if your webcam isn't at 0
CONF_THRESHOLD = 0.70       # ignore predictions below this
STABILITY_FRAMES = 8        # consecutive matching frames before locking on
PREDICT_EVERY_N = 2         # run YOLO every N frames for performance

# ---------- Globals shared between threads ----------
app = Flask(__name__)
model = YOLO(MODEL_PATH)
class_names = model.names   # e.g. {0: 'arcanine', 1: 'bulbasaur', ...}

latest_frame_jpeg = None    # most recent webcam frame as JPEG bytes
frame_lock = threading.Lock()

current_pokemon = None      # the name we've locked on to
current_confidence = 0.0
pokemon_info = None         # cached info dict for the current lock
recent_predictions = deque(maxlen=STABILITY_FRAMES)
api_cache = {}              # pokemon name -> info dict


# ---------- PokéAPI ----------
def fetch_pokemon_info(name: str):
    """Fetch name, genus, types, abilities, sprite, and flavor text from PokéAPI."""
    name = name.lower()
    if name in api_cache:
        return api_cache[name]

    try:
        # Main endpoint: types, abilities, sprite
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{name}", timeout=5)
        r.raise_for_status()
        data = r.json()

        types = [t["type"]["name"] for t in data["types"]]
        abilities = [a["ability"]["name"].replace("-", " ").title()
                     for a in data["abilities"]]
        sprite = (data["sprites"]["other"]["official-artwork"]["front_default"]
                  or data["sprites"]["front_default"])

        # Species endpoint: genus and flavor text
        r2 = requests.get(f"https://pokeapi.co/api/v2/pokemon-species/{name}", timeout=5)
        r2.raise_for_status()
        species = r2.json()

        # Flavor text: find the first English entry, clean up formatting chars
        flavor = ""
        for entry in species["flavor_text_entries"]:
            if entry["language"]["name"] == "en":
                flavor = (entry["flavor_text"]
                          .replace("\n", " ")
                          .replace("\f", " ")
                          .replace("  ", " "))
                break

        # Genus: "Flame Pokémon", "Mouse Pokémon", etc.
        genus = ""
        for g in species["genera"]:
            if g["language"]["name"] == "en":
                genus = g["genus"]
                break

        info = {
            "name": name.title(),
            "genus": genus,
            "types": types,
            "abilities": abilities,
            "sprite": sprite,
            "flavor": flavor,
        }
        api_cache[name] = info
        return info

    except Exception as e:
        print(f"[PokéAPI error] {name}: {e}")
        return {
            "name": name.title(),
            "genus": "",
            "types": [],
            "abilities": [],
            "sprite": "",
            "flavor": "No data available.",
        }


# ---------- Capture + inference thread ----------
def capture_loop():
    global latest_frame_jpeg, current_pokemon, current_confidence, pokemon_info

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # DSHOW = faster on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Try changing CAMERA_INDEX.")
        return

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame_count += 1

        # Run inference every N frames
        if frame_count % PREDICT_EVERY_N == 0:
            results = model.predict(frame, imgsz=224, verbose=False)
            r = results[0]
            top1_idx = int(r.probs.top1)
            conf = float(r.probs.top1conf)
            name = class_names[top1_idx]

            # Add to stability buffer (or None if below threshold)
            if conf >= CONF_THRESHOLD:
                recent_predictions.append(name)
            else:
                recent_predictions.append(None)

            # Lock-on check: all N recent predictions must agree
            if len(recent_predictions) == STABILITY_FRAMES:
                first = recent_predictions[0]
                if first is not None and all(p == first for p in recent_predictions):
                    if first != current_pokemon:
                        current_pokemon = first
                        current_confidence = conf
                        pokemon_info = fetch_pokemon_info(first)
                        print(f"[Locked on] {first} ({conf:.2f})")
                    else:
                        current_confidence = conf
                elif all(p is None for p in recent_predictions):
                    # Nothing confident for a while -> clear
                    if current_pokemon is not None:
                        print("[Cleared]")
                    current_pokemon = None
                    pokemon_info = None
                    current_confidence = 0.0

        # Encode frame as JPEG for streaming
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with frame_lock:
                latest_frame_jpeg = jpeg.tobytes()


# ---------- Flask routes ----------
def mjpeg_generator():
    """Yields webcam frames in MJPEG format for the <img> tag."""
    while True:
        with frame_lock:
            frame = latest_frame_jpeg
        if frame is None:
            time.sleep(0.03)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)  # ~30 fps cap


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/current")
def current():
    """JSON endpoint the front-end polls to update the info panel."""
    return jsonify({
        "pokemon": current_pokemon,
        "confidence": round(current_confidence, 2),
        "info": pokemon_info,
    })


# ---------- Entry point ----------
if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)