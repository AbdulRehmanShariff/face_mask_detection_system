
# """
# Face Mask Detection System
# Author: Rehman Shariff
# Built: 2026
# Tech: MobileNetV2 + OpenCV DNN + Flask
# """

# import os
# import cv2
# import numpy as np
# import base64
# import time
# from flask import Flask, render_template, request, jsonify, Response
# # from tensorflow.keras.models import load_model
# import tf_keras as keras_compat
# from tf_keras.models import load_model
# from werkzeug.utils import secure_filename

# IMG_SIZE = 224
# UPLOAD_FOLDER = "static/uploads"
# ALLOWED_EXT = {"jpg", "jpeg", "png"}

# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load model
# mask_model = None
# face_net = None

# def load_resources():
#     global mask_model, face_net
#     try:
#         # mask_model = load_model("models/best_model.keras")
#         mask_model = load_model("models/best_model_final.h5")
#         print("Model loaded successfully")
#     except Exception as e:
#         print(f"Model load error: {e}")

#     try:
#         face_net = cv2.dnn.readNet("deploy.prototxt", "face_detector.caffemodel")
#         print("Face detector loaded successfully")
#     except Exception as e:
#         print(f"Face detector load error: {e}")

# load_resources()

# stats = {"total": 0, "mask": 0, "nomask": 0, "start": time.time()}

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# def detect_faces(image):
#     if face_net is None:
#         return []
#     h, w = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(
#         cv2.resize(image, (300, 300)), 1.0,
#         (300, 300), (104.0, 177.0, 123.0)
#     )
#     face_net.setInput(blob)
#     detections = face_net.forward()
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             x1, y1, x2, y2 = box.astype("int")
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)
#             faces.append((x1, y1, x2, y2))
#     return faces

# def predict_mask(face_img):
#     if mask_model is None:
#         return "Unknown", 0.0, "#6b7fa3", "unknown"
#     rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#     resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
#     pred = mask_model.predict(np.expand_dims(resized, 0), verbose=0)[0][0]
#     if pred < 0.5:
#         return "With Mask", round(float((1 - pred) * 100), 2), "#34d399", "safe"
#     return "No Mask", round(float(pred * 100), 2), "#f87171", "danger"

# def process_image(image):
#     faces = detect_faces(image)
#     results = []
#     for (x1, y1, x2, y2) in faces:
#         face_img = image[y1:y2, x1:x2]
#         if face_img.size == 0:
#             continue
#         label, conf, hex_color, status = predict_mask(face_img)
#         hc = hex_color.lstrip("#")
#         r, g, b = [int(hc[i:i+2], 16) for i in (0, 2, 4)]
#         bgr = (b, g, r)
#         cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 3)
#         text = f"{label}: {conf}%"
#         (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
#         cv2.rectangle(image, (x1, y1 - th - 14), (x1 + tw + 12, y1), bgr, -1)
#         cv2.putText(image, text, (x1 + 6, y1 - 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
#         results.append({
#             "label": label,
#             "confidence": conf,
#             "color": hex_color,
#             "status": status
#         })
#     return results, image

# def generate_frames():
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.flip(frame, 1)
#         faces = detect_faces(frame)
#         mask_count = 0
#         nomask_count = 0
#         for (x1, y1, x2, y2) in faces:
#             face_img = frame[y1:y2, x1:x2]
#             if face_img.size == 0:
#                 continue
#             label, conf, hex_color, status = predict_mask(face_img)
#             if label == "With Mask":
#                 mask_count += 1
#             else:
#                 nomask_count += 1
#             hc = hex_color.lstrip("#")
#             r, g, b = [int(hc[i:i+2], 16) for i in (0, 2, 4)]
#             bgr = (b, g, r)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
#             txt = f"{label}: {conf}%"
#             (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#             cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), bgr, -1)
#             cv2.putText(frame, txt, (x1 + 5, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         h, w = frame.shape[:2]
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 20), -1)
#         cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
#         cv2.putText(frame, f"Faces: {len(faces)}", (15, 32),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2)
#         cv2.putText(frame, f"Mask: {mask_count}", (160, 32),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.75, (52, 211, 153), 2)
#         cv2.putText(frame, f"No Mask: {nomask_count}", (310, 32),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.75, (248, 113, 113), 2)
#         _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
#                + buffer.tobytes() + b"\r\n")
#     cap.release()


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/video_feed")
# def video_feed():
#     return Response(
#         generate_frames(),
#         mimetype="multipart/x-mixed-replace; boundary=frame"
#     )


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]

#     if not file.filename or not allowed_file(file.filename):
#         return jsonify({"error": "Only JPG and PNG files are allowed"}), 400

#     filename = secure_filename(file.filename)
#     save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     file.save(save_path)

#     image = cv2.imread(save_path)
#     if image is None:
#         return jsonify({"error": "Could not read image"}), 400

#     results, annotated = process_image(image)

#     result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_" + filename)
#     cv2.imwrite(result_path, annotated)

#     with open(result_path, "rb") as f:
#         img_base64 = base64.b64encode(f.read()).decode()

#     if not results:
#         return jsonify({
#             "error": "No face detected in this image",
#             "image": img_base64,
#             "results": [],
#             "total_faces": 0,
#             "mask_count": 0,
#             "nomask_count": 0
#         })

#     mask_count = sum(1 for r in results if r["label"] == "With Mask")
#     nomask_count = sum(1 for r in results if r["label"] == "No Mask")

#     stats["total"] += len(results)
#     stats["mask"] += mask_count
#     stats["nomask"] += nomask_count

#     return jsonify({
#         "image": img_base64,
#         "results": results,
#         "total_faces": len(results),
#         "mask_count": mask_count,
#         "nomask_count": nomask_count
#     })


# @app.route("/stats")
# def get_stats():
#     return jsonify({
#         **stats,
#         "uptime": int(time.time() - stats["start"])
#     })


# # if __name__ == "__main__":
# #     app.run(host="0.0.0.0", debug=False)
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=False)



"""
Face Mask Detection System
Author: Rehman Shariff
Built: 2026
Tech: MobileNetV2 + OpenCV DNN + Flask
"""

import os
import cv2
import numpy as np
import base64
import time
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

IMG_SIZE = 224
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mask_model = None
face_net = None

def load_resources():
    global mask_model, face_net
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras import layers, models

        base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        base.trainable = False
        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        mask_model = models.Model(inputs=base.input, outputs=output)

        weights = np.load("models/weights.npy", allow_pickle=True)
        mask_model.set_weights(list(weights))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model load error: {e}")

    try:
        face_net = cv2.dnn.readNet("deploy.prototxt", "face_detector.caffemodel")
        print("Face detector loaded successfully")
    except Exception as e:
        print(f"Face detector load error: {e}")

load_resources()

stats = {"total": 0, "mask": 0, "nomask": 0, "start": time.time()}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def detect_faces(image):
    if face_net is None:
        return []
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2, y2))
    return faces

def predict_mask(face_img):
    if mask_model is None:
        return "Unknown", 0.0, "#6b7fa3", "unknown"
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    pred = mask_model.predict(np.expand_dims(resized, 0), verbose=0)[0][0]
    if pred < 0.5:
        return "With Mask", round(float((1 - pred) * 100), 2), "#34d399", "safe"
    return "No Mask", round(float(pred * 100), 2), "#f87171", "danger"

def process_image(image):
    faces = detect_faces(image)
    results = []
    for (x1, y1, x2, y2) in faces:
        face_img = image[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        label, conf, hex_color, status = predict_mask(face_img)
        hc = hex_color.lstrip("#")
        r, g, b = [int(hc[i:i+2], 16) for i in (0, 2, 4)]
        bgr = (b, g, r)
        cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 3)
        text = f"{label}: {conf}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(image, (x1, y1 - th - 14), (x1 + tw + 12, y1), bgr, -1)
        cv2.putText(image, text, (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        results.append({
            "label": label,
            "confidence": conf,
            "color": hex_color,
            "status": status
        })
    return results, image

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        faces = detect_faces(frame)
        mask_count = 0
        nomask_count = 0
        for (x1, y1, x2, y2) in faces:
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            label, conf, hex_color, status = predict_mask(face_img)
            if label == "With Mask":
                mask_count += 1
            else:
                nomask_count += 1
            hc = hex_color.lstrip("#")
            r, g, b = [int(hc[i:i+2], 16) for i in (0, 2, 4)]
            bgr = (b, g, r)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
            txt = f"{label}: {conf}%"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), bgr, -1)
            cv2.putText(frame, txt, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, f"Faces: {len(faces)}", (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2)
        cv2.putText(frame, f"Mask: {mask_count}", (160, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (52, 211, 153), 2)
        cv2.putText(frame, f"No Mask: {nomask_count}", (310, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (248, 113, 113), 2)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")
    cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Only JPG and PNG files are allowed"}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
    image = cv2.imread(save_path)
    if image is None:
        return jsonify({"error": "Could not read image"}), 400
    results, annotated = process_image(image)
    result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_" + filename)
    cv2.imwrite(result_path, annotated)
    with open(result_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    if not results:
        return jsonify({
            "error": "No face detected in this image",
            "image": img_base64,
            "results": [],
            "total_faces": 0,
            "mask_count": 0,
            "nomask_count": 0
        })
    mask_count = sum(1 for r in results if r["label"] == "With Mask")
    nomask_count = sum(1 for r in results if r["label"] == "No Mask")
    stats["total"] += len(results)
    stats["mask"] += mask_count
    stats["nomask"] += nomask_count
    return jsonify({
        "image": img_base64,
        "results": results,
        "total_faces": len(results),
        "mask_count": mask_count,
        "nomask_count": nomask_count
    })


@app.route("/stats")
def get_stats():
    return jsonify({
        **stats,
        "uptime": int(time.time() - stats["start"])
    })

@app.route('/ping')
def ping():
    return "pong", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)