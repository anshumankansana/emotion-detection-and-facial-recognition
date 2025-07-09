import cv2
import dlib
import face_recognition
import pickle
import os
import numpy as np
from scipy.spatial import distance
import pyttsx3
from fer import FER
from datetime import datetime
import speech_recognition as sr
import openai
from dotenv import load_dotenv

# === Load API Key ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Setup Text-to-Speech ===
engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('volume', 1.0)
# Optional: Set to female voice if available
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)

# === Setup FER (deep learning model) ===
emotion_detector = FER()

# === Thresholds ===
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.75
TIRED_FRAMES = 15
tired_counter = 0

# === Voice Input (Supports Hindi & English) ===
def listen_to_user():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("🎧 Listening in Hindi or English...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=7)
        except sr.WaitTimeoutError:
            print("⏱️ You didn’t say anything in time.")
            return "You were silent, I didn’t hear anything."

    try:
        # Try Hindi first
        text = recognizer.recognize_google(audio, language='hi-IN')
        print(f"🗣️ You (Hindi): {text}")
        return text
    except sr.UnknownValueError:
        pass

    try:
        # Try English fallback
        text = recognizer.recognize_google(audio, language='en-IN')
        print(f"🗣️ You (English): {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that."
    except sr.RequestError:
        return "Speech recognition service is unavailable."

# === Emotion + tiredness logic ===
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# === GPT-Based Personality Reply ===
def get_personality_reply(name, emotion, history, tired=False):
    if name not in history:
        history[name] = {}

    prompt = f"""You are a sarcastic, smart, and emotional AI assistant named Robot Girl.
You recognize the person named {name}. Their current emotion is \"{emotion}\".
{'They look tired too.' if tired else ''}

Reply like a cute, witty robot girl. Be natural, funny, and personal.
Only return your one-line response."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a witty assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT error:", e)
        return f"{name}, I can't think of anything witty right now."

# === Load Memory ===
if os.path.exists("face_data.pkl"):
    with open("face_data.pkl", "rb") as f:
        known_faces, known_names = pickle.load(f)
else:
    known_faces, known_names = [], []

if os.path.exists("emotion_history.pkl"):
    with open("emotion_history.pkl", "rb") as f:
        emotion_history = pickle.load(f)
else:
    emotion_history = {}

# === Dlib & Video Setup ===
video = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("🧠 Smart Robot Girl AI with GPT Brain is running...")

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_faces, encoding)

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]
        else:
            cv2.imshow("Face", frame)
            print("👤 New face detected!")
            name = input("Enter name: ")
            known_faces.append(encoding)
            known_names.append(name)
            emotion_history[name] = {}

        # Landmark-based tiredness check
        rects = detector(gray, 0)
        tired_now = False
        for rect in rects:
            shape = predictor(gray, rect)
            shape_np = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            left_eye = shape_np[36:42]
            right_eye = shape_np[42:48]
            mouth = shape_np[48:68]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
                tired_counter += 1
            else:
                tired_counter = 0

            if tired_counter >= TIRED_FRAMES:
                tired_now = True

            break

        # FER-based emotion detection
        detected_emotion = emotion_detector.top_emotion(frame)
        mood = detected_emotion[0] if detected_emotion else "neutral"

        # Memory & reply
        emotion_history[name][mood] = emotion_history[name].get(mood, 0) + 1
        reply = get_personality_reply(name, mood, emotion_history, tired_now)

        # Show and speak
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 100, 100), 2)
        cv2.putText(frame, f"{name}: {mood}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"💬 {reply}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        engine.say(reply)
        engine.runAndWait()

        # Listen to user reply in Hindi or English
        user_input = listen_to_user()
        engine.say(f"You said: {user_input}")
        engine.runAndWait()

    cv2.imshow("Robot Girl - Merged AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open("emotion_history.pkl", "wb") as f:
    pickle.dump(emotion_history, f)

video.release()
cv2.destroyAllWindows()