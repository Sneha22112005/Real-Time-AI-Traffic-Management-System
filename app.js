from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Allowed vehicle classes
vehicle_ids = [2, 3, 5, 7]   # car, motorbike, bus, truck


@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None
    count = 0

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        # Save file
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Read the image
        frame = cv2.imread(filepath)

        # Run YOLO detection
        results = model.predict(frame, conf=0.4, classes=vehicle_ids, verbose=False)
        boxes = results[0].boxes
        count = len(boxes)

        # Draw detection boxes
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save output image
        output_filename = f"output_{uuid.uuid4()}.jpg"
        output_path = os.path.join("static", output_filename)
        cv2.imwrite(output_path, frame)

        output_image = output_filename

    return render_template("index.html", output_image=output_image, count=count)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == "__main__":
    app.run(debug=True)
