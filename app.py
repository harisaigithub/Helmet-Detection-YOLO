import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

# Function to download files
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# URLs of the files hosted externally
file_urls = {
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3.cfg": "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg",
    "coco.names": "https://github.com/pjreddie/darknet/raw/master/data/coco.names"
}

# Download the files if not present
for filename, url in file_urls.items():
    if not os.path.exists(filename):
        st.info(f"Downloading {filename}...")
        download_file(url, filename)
        st.success(f"Downloaded {filename}!")

# Sidebar content
st.sidebar.image('HEMANTH TULIMILLI.png', use_container_width=True)
st.sidebar.header("**HEMANTH TULIMILLI**")
st.sidebar.write("Expert in the IOT Domain")

st.sidebar.header("About Model")
st.sidebar.info('''This Model is designed for real-time helmet detection using the YOLOv3 (You Only Look Once) model.
    1️⃣ Click on Upload button.
    2️⃣ Upload images to detect helmets.
    3️⃣ See the results.''')
st.sidebar.header("Contact Information")
st.sidebar.write("[GitHub](https://github.com/Hemanthtu/)")
st.sidebar.write("[Email](mailto:hemanthtulimilli.18@gmail.com)")
st.sidebar.write("Developed by Hemanth Tulimilli with guidance of Dept.IOT KLU.")

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Ensure "helmet" is part of the labels
if "helmet" not in classes:
    st.warning("The default COCO dataset does not include 'helmet' as a class. Consider using a custom-trained YOLO model.")

# Function to detect objects
def detect_objects(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape

    # Convert image to blob format (required by YOLO)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Iterate over detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # If confidence is above the threshold
            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform Non-Maximum Suppression to eliminate overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Adjust NMS threshold
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Draw bounding boxes and labels
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_label = f"{confidences[i]:.2f}"
            color = colors[class_ids[i]]

            # Check if the detected object is "helmet"
            if label.lower() == "helmet":
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label}: {confidence_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# Main application
st.title("YOLO Helmet Detection")
st.write("Upload an image to detect helmets.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_container_width=True)
    st.write("Detecting...")
    result_image = detect_objects(image)
    st.image(result_image, caption="Processed Image.", use_container_width=True)
