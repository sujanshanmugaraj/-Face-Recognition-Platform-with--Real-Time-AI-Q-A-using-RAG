from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import sqlite3
import base64
import io
from PIL import Image
from datetime import datetime
from recognition_stream import recognition_bp


app = Flask(__name__)
CORS(app)  # Allow requests from frontend

DB_PATH = 'face_data.db'  # Path to SQLite database

# Create table if not exists
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    encoding BLOB,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return "Flask API is running"

@app.route('/register_face', methods=['POST'])
def register_face():
    data = request.json
    name = data['name']
    image_data = data['image']

    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)

    # Get face encodings
    encodings = face_recognition.face_encodings(image_np)
    if not encodings:
        return jsonify({'status': 'error', 'message': 'No face detected!'}), 400

    encoding = encodings[0]
    timestamp = datetime.now().isoformat()

    # Store in SQLite DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO faces (name, encoding, timestamp) VALUES (?, ?, ?)",
              (name, encoding.tobytes(), timestamp))
    conn.commit()
    conn.close()

    return jsonify({'status': 'success', 'message': f'{name} registered at {timestamp}'})

if __name__ == '__main__':
    init_db()  # Initialize the database
    app.register_blueprint(recognition_bp)

    app.run(host='0.0.0.0', port=5000)
