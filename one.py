import cv2
import numpy as np
import pyttsx3
from keras.src.models import load_model

# Load trained currency detection model
model = load_model('model/currency_classifier.h5')

# Load label names (e.g., 10, 20, 50, ...)
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Start webcam
cap = cv2.VideoCapture(0)
speak("Camera started. Show the currency note.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize and preprocess the image
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)

    # Predict currency
    predictions = model.predict(img)
    index = np.argmax(predictions)
    currency = labels[index]
    confidence = predictions[0][index]

    # Show result
    cv2.putText(frame, f"{currency} Rupees ({confidence*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Currency Detection", frame)

    # Speak once if confident
    if confidence > 0.9:
        speak(f"This is {currency} rupees")
        cv2.waitKey(3000)  # wait to avoid repeating quickly

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        speak("Exiting currency detector.")
        break

cap.release()
cv2.destroyAllWindows()
