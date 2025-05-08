import os
import cv2
import pickle
import face_recognition
import numpy as np
from datetime import datetime

ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'encodings.pkl')

def load_known_faces():
    if not os.path.exists(ENCODINGS_PATH):
        print("❌ No encodings found. Please register faces first.")
        return {}, []
    
    with open(ENCODINGS_PATH, 'rb') as f:
        data = pickle.load(f)
    
    names = list(data.keys())
    encodings = [data[name]['encoding'] for name in names]
    return dict(zip(names, encodings)), names

def recognize_faces():
    known_faces, names_list = load_known_faces()
    if not known_faces:
        return

    video = cv2.VideoCapture(0)
    print("[INFO] Starting live face recognition. Press 'q' to quit.")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            face_distances = face_recognition.face_distance(list(known_faces.values()), face_encoding)
            name = "Unknown"

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = names_list[best_match_index]

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow('Live Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
