import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyttsx3
import sys
import os

# Load trained model
model_path = "model/currency_model.h5"
if not os.path.exists(model_path):
    print("Model not found! Please place your trained model at model/currency_model.h5")
    sys.exit(1)

model = load_model(model_path)

# Class labels - update based on your trained model
labels = ['₹10', '₹20', '₹50', '₹100', '₹200', '₹500', '₹2000']

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def predict_currency(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_index = np.argmax(prediction[0])
        currency = labels[class_index]

        print(f"Detected currency: {currency}")
        speak(f"The currency note is {currency}")

    except Exception as e:
        print("Error:", e)
        speak("Sorry, I could not detect the currency.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_currency.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_currency(image_path)
