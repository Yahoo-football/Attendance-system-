from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import sqlite3
import io
from openpyxl import Workbook
from openpyxl.styles import Font

app = Flask(__name__)
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
            # Resize to 240px for speed
            img = cv2.resize(img, (240, 240))
            enc = face_recognition.face_encodings(img, num_jitters=1)
            if enc:
                encodings.append(enc[0])
                names.append(os.path.splitext(filename)[0])
    return encodings, names

# Cache known faces
if not hasattr(app, 'known_data'):
    app.known_data = load_known_faces()

# ==================== FAST SCAN (INSTANT NAME) ====================
@app.route('/scan', methods=['POST'])
def scan():
    known_encodings, known_names = app.known_data
    if not known_encodings:
        return jsonify({"attendances": []})

    # Read + resize FAST
    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    new_w = 240  # Ultra fast
    img = cv2.resize(img, (new_w, int(h * new_w / w)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Fast detection + encoding
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations, num_jitters=1)

    results = []
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    for (top, right, bottom, left), enc in zip(locations, encodings):
        # Fast match
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.55)
        distances = face_recognition.face_distance(known_encodings, enc)
        idx = np.argmin(distances)
        name = known_names[idx] if matches[idx] else "Unknown"

        # Scale back
        scale = w / new_w
        scaled_loc = [int(top*scale), int(right*scale), int(bottom*scale), int(left*scale)]

        # Check if already recorded today
        already_today = False
        if name != "Unknown":
            c.execute("SELECT 1 FROM attendance WHERE name=? AND DATE(timestamp)=?", (name, today_str))
            already_today = c.fetchone() is not None

        # Save only once per day
        if name != "Unknown" and not already_today:
            ts = now.strftime('%Y-%m-%d %H:%M:%S')
            c.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, ts))
            conn.commit()

        results.append({
            "name": name,
            "location": scaled_loc,
            "already_today": already_today
        })

    conn.close()
    return jsonify({"attendances": results})

# ==================== FAST UPLOAD ====================
@app.route('/upload_face', methods=['POST'])
def upload_face():
    name = request.form['name'].strip()
    file = request.files['file']
    if not name or not file:
        return jsonify({"error": "Name and photo required"})

    # Resize to 240px for speed
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (240, 240))
    _, buffer = cv2.imencode('.jpg', img)
    file.stream = io.BytesIO(buffer)

    filename = f"{name}.jpg"
    filepath = os.path.join(PHOTOS_DIR, filename)
    with open(filepath, 'wb') as f:
        f.write(buffer)

    # Reload known faces
    app.known_data = load_known_faces()
    return jsonify({"message": f"Registered {name}"})

# ==================== RECORDS ====================
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

# ==================== EXCEL EXPORT ====================
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

# ==================== HOME ====================
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)