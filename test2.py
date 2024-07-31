import cv2
import numpy as np
import time
import argparse

# Argument parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--input', type=str, default='0', help='Input source. 0 for webcam or video file path.')
args = parser.parse_args()

# Load the pre-trained neural network model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Set the minimum confidence threshold from arguments
confidence_threshold = args.confidence

# Check if input is webcam or video file
if args.input == '0':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.input)

# Initialize variables for FPS calculation
fps_start_time = 0
fps = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is not captured, break the loop
    if not ret:
        break

    # Get the frame dimensions
    (h, w) = frame.shape[:2]

    # Preprocess the frame: resize to 300x300 pixels and normalize
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the input to the neural network
    net.setInput(blob)

    # Perform object detection
    detections = net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence threshold
        if confidence > confidence_threshold:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and display the confidence
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Calculate FPS
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / time_diff
    fps_start_time = fps_end_time

    # Display FPS on frame
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Frame", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
