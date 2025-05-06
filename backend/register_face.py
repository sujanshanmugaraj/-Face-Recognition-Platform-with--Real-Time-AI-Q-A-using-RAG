import cv2
import face_recognition
import numpy as np
import pickle
import os
from datetime import datetime
from database import Session, FaceEntry

# Directory & file setup
ENCODINGS_PATH = "../data/encodings.pkl"
LOG_PATH = "./logs/events.log"
os.makedirs("../data", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Load existing encodings
if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, "rb") as f:
        known_encodings = pickle.load(f)
else:
    known_encodings = []

# Start webcam
video = cv2.VideoCapture(0)
print("[INFO] Press 's' to scan face and save, 'q' to quit.")

while True:
    ret, frame = video.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    for top, right, bottom, left in face_locations:
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Face Registration", frame)
    key = cv2.waitKey(1)

    if key == ord("s") and face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        name = input("Enter name: ").strip()

        if name and face_encodings:
            known_encodings.append({
                "name": name,
                "encoding": face_encodings[0].tolist(),
                "timestamp": str(datetime.now())
            })

            # Save to .pkl
            with open(ENCODINGS_PATH, "wb") as f:
                pickle.dump(known_encodings, f)

            # Save to DB
            session = Session()
            session.add(FaceEntry(name=name, timestamp=datetime.now()))
            session.commit()
            session.close()

            # Log event
            with open(LOG_PATH, "a") as log:
                log.write(f"[{datetime.now()}] Registered: {name}\n")

            print(f"[✓] Face of {name} registered successfully.")
        else:
            print("[!] Face not detected properly.")

    elif key == ord("q"):
        break


video.release()
cv2.destroyAllWindows()
