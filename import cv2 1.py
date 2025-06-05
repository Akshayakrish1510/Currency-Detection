import cv2
import numpy as np
from keras.models import load_model
import pyttsx3
from time import sleep

# Load the trained model
model = load_model('model/currency_classifier.h5')

# Load labels
with open("labels.txt", "r") as file:
    labels = [line.strip() for line in file.readlines()]

# Setup text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera. Please check your webcam.")
    speak("Unable to access camera.")
    exit()

print("✅ Camera started. Press 'q' to quit.")
speak("Currency detection started. Hold the note in front of the camera.")

last_label = ""
cooldown = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    # Preprocessing
    resized = cv2.resize(frame, (224, 224))
    img = resized.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)
    label_index = np.argmax(pred)
    label = labels[label_index]
    confidence = pred[0][label_index]

    # Display
    text = f"{label} Rupees ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)
    cv2.imshow("Currency Detection", frame)

    # Speak if new label with high confidence
    if confidence > 0.85 and label != last_label and cooldown == 0:
        speak(f"This is {label} rupees")
        last_label = label
        cooldown = 50  # Number of frames to wait before repeating

    if cooldown > 0:
        cooldown -= 1

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        speak("Closing the application.")
        break

cap.release()
cv2.destroyAllWindows()
