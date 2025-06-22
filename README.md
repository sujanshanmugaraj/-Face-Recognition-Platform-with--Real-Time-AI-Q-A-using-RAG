## 🧠 Face Recognition Platform with Real-Time AI Q\&A using RAG

A full-stack AI platform that performs **face registration**, **real-time recognition**, and **context-aware question answering** using **RAG (Retrieval-Augmented Generation)**.

---

### 🔍 Detailed Overview

This project includes 3 main components:

#### 1. Face Registration

* Uses either webcam or uploaded image
* Extracts face encodings using `face_recognition`
* Stores them in `data/encodings.pkl`
* Logs activity in `logs/events.log`

#### 2. Real-Time Face Recognition

* Uses webcam input via OpenCV
* Compares faces against encodings database
* Draws bounding boxes and labels recognized faces live
* Optional: returns name and confidence score via `/recognize` endpoint

#### 3. RAG (Retrieval-Augmented Generation) Module *(planned)*

* Accepts questions from a user
* Searches relevant documents or face logs
* Uses a language model (e.g., GPT-4 via OpenAI API) to answer
* Will be served via a `/ask` API endpoint

---

### 📁 Folder Structure

```
backend/
├── app.py                     # Flask server
├── requirements.txt
├── logs/
│   └── events.log             # Face registration logs
├── data/
│   ├── encodings.pkl          # Saved face encodings
│   └── faces/                 # Temporarily saved face images
├── routes/
│   ├── __init__.py
│   ├── register.py            # /register endpoint
│   └── recognize.py           # /recognize endpoint (coming soon)
├── utils/
│   ├── __init__.py
│   ├── encodings.py           # Load/save pickle encodings
│   ├── logger.py              # Log registration info
│   ├── database.py            # Optional DB handling (if used)
│   └── rag_engine.py          # Placeholder for AI Q&A (if using OpenAI/LangChain)
```

---

### 🚀 Features

* ✅ Face registration using webcam or image upload
* ✅ Real-time face recognition (camera or video feed)
* 🧠 RAG pipeline (for answering user questions based on retrieved context)
* 📆 Encodings stored in `pickle` file
* 📜 Registration logs with timestamp
* 🧩 Modular Flask API using Blueprints

---

### ⚙️ Setup Instructions

1. **Clone the repo** and `cd` into the backend directory.

2. **Create virtual environment:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server:**

   ```bash
   python app.py
   ```

   You'll see:

   ```
   * Running on http://127.0.0.1:5000/
   ```

---

### 📄 Register Face via API

**POST** `/register`
Use Postman or curl:

```bash
curl -X POST -F "name=Sujan" -F "image=@/path/to/image.jpg" http://127.0.0.1:5000/register
```

**Success Response:**

```json
{ "message": "Face for 'Sujan' registered successfully!" }
```

---


### 📡 API Endpoints Summary

#### `/` (GET)

Returns: *"Face Recognition API is running!"*

#### `/register` (POST)

* Accepts `name` and `image` via `form-data`
* Saves encoding and logs the registration
* Returns JSON with success message or error

#### `/recognize` *(Planned)*

* Accepts live frame or image
* Returns matched name if found, otherwise "Unknown"

#### `/ask` *(Planned)*

* Accepts user query (and optionally face ID)
* Retrieves context + generates response using RAG engine

---

### 🔧 Tech Stack

* Python 3.10+
* Flask
* OpenCV
* face\_recognition
* numpy
* werkzeug
* (Planned: LangChain, OpenAI API)

---

### 🧪 Testing Notes

* Use Postman for image upload
* Check `data/encodings.pkl` after registration
* Log details can be monitored via `logs/events.log`

