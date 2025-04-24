import cv2
import mediapipe as mp
import pyttsx3
import time
import sys
import numpy as np
from deepface import DeepFace

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

prev_text = ""
recognized_sentence = []
last_gesture_time = time.time()
gesture_timeout = 2.0  # Timeout before finalizing a phrase

# Hand Gesture-to-Sentence Mapping
gesture_mappings = {
    "OPEN_PALM": "Hello! How are you feeling today?",
    "FIST": "I need help, can you assist me?",
    "THUMBS_UP": "Yes, that sounds great!",
    "THUMBS_DOWN": "No, I don't agree.",
    "VICTORY": "Thank you very much!",
    "POINTING_UP": "What is your name?",
    "HAND_WAVE": "Goodbye! Have a nice day.",
}

# Function to detect predefined hand gestures
def detect_hand_gesture(landmarks):
    """Detects hand gestures based on finger positions."""
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y

    if index_tip < middle_tip < ring_tip < pinky_tip:  
        return "OPEN_PALM"  # ✋ Hello
    elif thumb_tip > index_tip > middle_tip > ring_tip > pinky_tip:  
        return "FIST"  # ✊ I need help
    elif thumb_tip < index_tip and middle_tip < ring_tip < pinky_tip:  
        return "THUMBS_UP"  # 👍 Yes
    elif thumb_tip > index_tip and middle_tip > ring_tip > pinky_tip:  
        return "THUMBS_DOWN"  # 👎 No
    elif index_tip < middle_tip and ring_tip < pinky_tip:  
        return "VICTORY"  # ✌️ Thank you
    elif index_tip < thumb_tip and middle_tip > ring_tip > pinky_tip:  
        return "POINTING_UP"  # ☝️ What is your name?
    elif index_tip < middle_tip and ring_tip > pinky_tip and thumb_tip < index_tip:
        return "HAND_WAVE"  # 👋 Goodbye
    
    return None

# Function to detect facial expressions
def detect_facial_expression(frame):
    """Uses DeepFace to detect emotions from a face."""
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion.capitalize()  # E.g., 'Happy', 'Sad'
    except:
        return "Neutral"

# Function to generate output sentence from hand + face
def generate_output(gesture, expression):
    """Generates a full sentence based on gesture and facial expression."""
    if gesture in gesture_mappings:
        sentence = gesture_mappings[gesture]
    else:
        sentence = "I am here to assist you."

    if expression == "Happy":
        sentence += " You seem happy! That's great to see!"
    elif expression == "Sad":
        sentence += " You look sad. Do you need someone to talk to?"
    elif expression == "Angry":
        sentence += " I sense frustration. Can I help with something?"
    elif expression == "Surprise":
        sentence += " You look surprised! Is everything okay?"
    else:
        sentence += " How can I help you today?"

    return sentence

# Function to finalize and speak the sentence
def speak_text(sentence):
    """Speaks out the generated sentence."""
    global prev_text, recognized_sentence
    if sentence and sentence != prev_text:
        print("\n🗣️ Speaking:", sentence)
        engine.say(sentence)
        engine.runAndWait()
        prev_text = sentence

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_detected = None
    face_expression = "Neutral"

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            gesture_detected = detect_hand_gesture(landmarks.landmark)
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Detect facial expression
    face_expression = detect_facial_expression(frame)

    if gesture_detected:
        current_time = time.time()
        if current_time - last_gesture_time > gesture_timeout:
            sentence = generate_output(gesture_detected, face_expression)
            speak_text(sentence)
        last_gesture_time = current_time

        # Display detected gesture and expression
        cv2.putText(frame, f'Gesture: {gesture_detected}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Face: {face_expression}', (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Gesture & Expression Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()