import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model(
    "face_mask_model.h5"
)

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Webcam
camera = cv2.VideoCapture(0)

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            # Resize for MobileNetV2
            face = cv2.resize(
                face,
                (224, 224)
            )

            face = face.astype("float32")
            face = face / 255.0

            face = np.expand_dims(
                face,
                axis=0
            )

            # Predict
            prediction = model.predict(
                face,
                verbose=0
            )[0][0]

            # IMPORTANT:
            # 0 = with mask
            # 1 = without mask

            if prediction < 0.5:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                color,
                2
            )

            # Show label
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        ret, buffer = cv2.imencode(
            '.jpg',
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )

@app.route('/')
def index():
    return render_template(
        'index.html'
    )

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype=
        'multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(debug=True)