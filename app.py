from flask import Flask, render_template, Response, request, jsonify
from flask_pymongo import PyMongo
import cv2
import uuid
import os
import datetime
import threading
from imutils import paths
import face_recognition
import pickle
import time

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/user_db"
mongo = PyMongo(app)

# Ensure dataset directory exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

camera = cv2.VideoCapture(0)
capture_images = False
current_user_id = None

def capture_frames():
    global capture_images, current_user_id
    while capture_images and current_user_id:
        success, frame = camera.read()
        if success:
            img_name = f'dataset/{current_user_id}/image_{uuid.uuid4().hex}.jpg'
            cv2.imwrite(img_name, frame)
            time.sleep(1)
            print(f"Captured {img_name}")

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/user', methods=['POST'])
def create_user():
    global current_user_id
    first_name = request.form.get('firstName')
    last_name = request.form.get('lastName')
    email = request.form.get('email')
    mobile = request.form.get('mobile')

    user_info = {
        "firstName": first_name,
        "lastName": last_name,
        "email": email,
        "mobile": mobile,
        "createdDate": datetime.datetime.utcnow(),
        "modifiedDate": datetime.datetime.utcnow(),
        "uniqueIdentifier": str(uuid.uuid4())
    }

    current_user_id = user_info["firstName"]
    if not os.path.exists(f'dataset/{current_user_id}'):
        os.makedirs(f'dataset/{current_user_id}')
    
    existing_user = mongo.db.users.find_one({"email": email})
    if not existing_user:
        mongo.db.users.insert_one(user_info)
        return jsonify({"message": "User created", "userId": current_user_id}), 201
    else:
        return jsonify({"message": "User already exists", "userId": existing_user["uniqueIdentifier"]}), 200

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capture_images
    capture_images = True
    threading.Thread(target=capture_frames).start()  # Start capturing in a separate thread
    return jsonify({"message": "Started capturing images"}), 200

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global capture_images
    capture_images = False
    return jsonify({"message": "Stopped capturing images"}), 200

@app.route('/train', methods=['POST'])
def train_dataset():
    encodingsP = "encodings.pickle"
    print("[INFO] start processing faces...")
    imagePaths = list(paths.list_images("dataset"))

    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open(encodingsP, "wb") as f:
        f.write(pickle.dumps(data))

    return jsonify({"message": "Training completed and encodings saved"}), 200

@app.route('/stop')
def stop():
    camera.release()
    return "Camera stopped", 200

if __name__ == "__main__":
    app.run(debug=True)
