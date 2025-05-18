# sim_rpi_flask.py
from flask import Flask, Response
import cv2

app = Flask(__name__)

def generate():
    cap = cv2.VideoCapture(0)  # Usa tu webcam local
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/live/')
def live():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', port=80, threaded=True)