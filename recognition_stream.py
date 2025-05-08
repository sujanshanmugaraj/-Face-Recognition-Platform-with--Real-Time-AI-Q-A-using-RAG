# recognition_stream.py
from flask import Response, Blueprint
import cv2
import face_recognition
import numpy as np
import sqlite3
import io

recognition_bp = Blueprint('recognition', __name__)

DB_PATH = 'face_data.db'

def load_known_faces():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM faces")
    data = c.fetchall()
    conn.close()

    known_names = []
    known_encodings = []

    for name, encoding in data:
        known_names.append(name)
        known_encodings.append(np.frombuffer(encoding, dtype=np.float64))

    return known_names, known_encodings


def gen_frames():
    known_names, known_encodings = load_known_faces()
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@recognition_bp.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
