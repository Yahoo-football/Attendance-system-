from flask import Flask, request, jsonify, render_template, send_file, url_for
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import sqlite3
import io
from openpyxl import Workbook
from openpyxl.styles import Font

app = Flask(__name__, static_folder='static')
DB_NAME = "attendance.db"
PHOTOS_DIR = "photos"

# Create folders and DB
os.makedirs(PHOTOS_DIR, exist_ok=True)
conn = sqlite3.connect(DB_NAME)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    timestamp TEXT NOT NULL
)''')
conn.commit()
conn.close()

# ==================== FAST LOAD KNOWN FACES ====================
def load_known_faces():
    encodings = []
    names = []
    if not os.path.exists(PHOTOS_DIR):
        return [], []

    for filename in os.listdir(PHOTOS_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(PHOTOS_DIR, filename)
            img = face_recognition.load_image_file(path)
            img = cv2.resize(img, (240, 240))
            enc = face_recognition.face_encodings(img, num_jitters=1)
            if enc:
                encodings.append(enc[0])
                names.append(os.path.splitext(filename)[0])
    return encodings, names

# Cache known faces
if not hasattr(app, 'known_data'):
    app.known_data = load_known_faces()

# ==================== ULTRA-FAST SCAN + INSTANT SOUND ====================
@app.route('/scan', methods=['POST'])
def scan():
    known_encodings, known_names = app.known_data
    if not known_encodings:
        return jsonify({"attendances": [], "play_sound": False})

    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    new_w = 160
    small = cv2.resize(img, (new_w, int(h * new_w / w)))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    model = "cnn" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "hog"
    locations = face_recognition.face_locations(rgb_small, model=model)
    encodings = face_recognition.face_encodings(rgb_small, locations, num_jitters=1)

    results = []
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    any_known = False

    for (top, right, bottom, left), enc in zip(locations, encodings):
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.55)
        distances = face_recognition.face_distance(known_encodings, enc)
        idx = np.argmin(distances)
        name = known_names[idx] if matches[idx] else "Unknown"

        scale = w / new_w
        scaled_loc = [int(v * scale) for v in (top, right, bottom, left)]

        already_today = False
        if name != "Unknown":
            c.execute("SELECT 1 FROM attendance WHERE name=? AND DATE(timestamp)=?", (name, today_str))
            already_today = c.fetchone() is not None

            if not already_today:
                ts = now.strftime('%Y-%m-%d %H:%M:%S')
                c.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, ts))
                conn.commit()

            any_known = True

        results.append({
            "name": name,
            "location": scaled_loc,
            "already_today": already_today
        })

    conn.close()
    return jsonify({"attendances": results, "play_sound": any_known})

# ==================== UPLOAD FACE ====================
@app.route('/upload_face', methods=['POST'])
def upload_face():
    name = request.form['name'].strip()
    file = request.files['file']
    if not name or not file:
        return jsonify({"error": "Name and photo required"})

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (240, 240))
    _, buffer = cv2.imencode('.jpg', img)

    filename = f"{name}.jpg"
    filepath = os.path.join(PHOTOS_DIR, filename)
    with open(filepath, 'wb') as f:
        f.write(buffer)

    app.known_data = load_known_faces()
    return jsonify({"message": f"Registered {name}"})

# ==================== RECORDS & EXPORT ====================
@app.route('/records')
def records():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, name, timestamp FROM attendance ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"id": r[0], "name": r[1], "time": r[2]} for r in rows])

@app.route('/delete_record/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Deleted"})

@app.route('/delete_all_records', methods=['DELETE'])
def delete_all_records():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()
    return jsonify({"message": "All deleted"})

@app.route('/export_csv')
def export_csv():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, timestamp FROM attendance ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return "No records", 400

    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"
    header = ['Name', 'Date & Time']
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for name, ts in rows:
        ws.append([name, ts])

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='attendance.xlsx'
    )

# ==================== PAGES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

# ==================== SHOW ALL REGISTERED FACES (NAMES + PHOTOS) ====================
@app.route('/known_faces')
def known_faces():
    names = [os.path.splitext(f)[0] for f in os.listdir(PHOTOS_DIR) 
             if f.lower().endswith(('.jpg','.jpeg','.png'))]
    return jsonify(sorted(names))

@app.route('/known_faces_page')
def known_faces_page():
    return render_template('known_faces.html')

if __name__ == '__main__':
    app.run(debug=True)