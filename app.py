# app.py
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from datetime import datetime
import sqlite3
import face_recognition

app = Flask(__name__)
PHOTO_DIR = "photos"
DB_NAME = "attendance.db"

# -------------------------------------------------
# DB Setup
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        match_percent REAL,
        auto_mark INTEGER DEFAULT 0
    )''')
    for col, sql in [
        ("match_percent REAL", "ALTER TABLE attendance ADD COLUMN match_percent REAL"),
        ("auto_mark INTEGER DEFAULT 0", "ALTER TABLE attendance ADD COLUMN auto_mark INTEGER DEFAULT 0")
    ]:
        try: c.execute(sql)
        except: pass
    conn.commit()
    conn.close()
init_db()

# -------------------------------------------------
# Load Known Faces (one encoding per photo)
# -------------------------------------------------
known_encodings = []
known_names = []

def load_known_faces():
    global known_encodings, known_names
    known_encodings, known_names = [], []
    if not os.path.isdir(PHOTO_DIR):
        os.makedirs(PHOTO_DIR)
        return
    for fn in os.listdir(PHOTO_DIR):
        if not fn.lower().endswith(('.png','.jpg','.jpeg')): continue
        path = os.path.join(PHOTO_DIR, fn)
        try:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if not encs:
                print(f"[WARN] No face detected in {fn}")
                continue
            known_encodings.append(encs[0])
            name = os.path.splitext(fn)[0].replace('_',' ').title()
            known_names.append(name)
            print(f"[LOAD] Loaded face: {name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {fn}: {e}")
load_known_faces()

# -------------------------------------------------
# ONE RECORD PER PERSON PER DAY
# -------------------------------------------------
def mark_attendance(name, match_percent=None, auto_mark=False):
    """Insert only if no record exists today. Returns timestamp (new or existing)."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Check if already marked today
    c.execute("SELECT timestamp FROM attendance WHERE name=? AND DATE(timestamp)=?", (name, today))
    row = c.fetchone()
    if row:
        conn.close()
        return row[0]  # Return existing timestamp
    
    # Insert new record
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        c.execute(
            "INSERT INTO attendance (name, timestamp, match_percent, auto_mark) VALUES (?,?,?,?)",
            (name, now, match_percent, 1 if auto_mark else 0)
        )
        conn.commit()
    except:
        c.execute("INSERT INTO attendance (name, timestamp) VALUES (?,?)", (name, now))
        conn.commit()
    conn.close()
    return now

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_face', methods=['POST'])
def upload_face():
    file = request.files['file']
    name = request.form.get('name','').strip()
    if not file or not name:
        return jsonify({"error":"Name + photo required"}), 400
    ext = os.path.splitext(file.filename)[1]
    fn = f"{name.replace(' ','_').lower()}{ext}"
    path = os.path.join(PHOTO_DIR, fn)
    file.save(path)
    load_known_faces()
    return jsonify({"message": f"Registered: {name}"})

@app.route('/delete_registered_face', methods=['POST'])
def delete_face():
    fn = request.get_json().get('filename')
    if not fn: return jsonify({"success":False,"error":"filename required"}), 400
    p = os.path.join(PHOTO_DIR, fn)
    if os.path.exists(p):
        os.remove(p)
        load_known_faces()
        return jsonify({"success":True})
    return jsonify({"success":False,"error":"not found"}), 404

@app.route('/scan', methods=['POST'])
def scan():
    """Scan uploaded image and return detected faces with names."""
    if not known_encodings:
        return jsonify({"message":"Register at least one face first."}), 400

    # Read uploaded image
    filestr = request.files['file'].read()
    npimg = np.frombuffer(filestr, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"message":"Bad image"}), 400

    # Detect faces
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    results = []
    for (top, right, bottom, left), enc in zip(locations, encodings):
        # Find best match
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.55)
        distances = face_recognition.face_distance(known_encodings, enc)
        best_idx = np.argmin(distances) if len(distances) else -1

        name = "Unknown"
        pct = 0.0
        auto = False
        if best_idx < len(matches) and matches[best_idx]:
            name = known_names[best_idx]
            pct = round((1 - distances[best_idx]) * 100, 1)
            auto = pct >= 70  # Auto-mark threshold

        # Mark attendance (only if new for today)
        ts = None
        already_today = False
        if name != "Unknown":
            ts = mark_attendance(name, pct, auto)
            already_today = datetime.now().strftime("%Y-%m-%d") in ts.split()[0]

        results.append({
            "name": name,
            "time": ts,
            "match_percent": pct,
            "auto_mark": auto,
            "already_today": already_today,
            "location": [top, right, bottom, left]  # For drawing box on face
        })

    return jsonify({"attendances": results})

@app.route('/records')
def records():
    """Get all attendance records."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, name, timestamp, match_percent, auto_mark FROM attendance ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "name": r[1],
            "time": r[2],
            "match_percent": float(r[3]) if r[3] is not None else None,
            "auto_mark": bool(r[4])
        })
    return jsonify(out)

@app.route('/delete_record/<int:rec_id>', methods=['DELETE'])
def delete_record(rec_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE id=?", (rec_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route('/delete_all_records', methods=['DELETE'])
def delete_all_records():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route('/export_csv')
def export_csv():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        WITH FirstAttendance AS (
            SELECT name, timestamp, DATE(timestamp) as attend_date,
                   ROW_NUMBER() OVER (PARTITION BY name, DATE(timestamp) ORDER BY timestamp) as rn
            FROM attendance
        )
        SELECT name, timestamp
        FROM FirstAttendance
        WHERE rn = 1
        ORDER BY timestamp DESC
    """)
    rows = c.fetchall()
    conn.close()

    import io, csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'First Check-in Time'])
    writer.writerows(rows)
    
    return output.getvalue(), 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=attendance.csv'
    }

if __name__ == '__main__':
    app.run(debug=True)