import cv2
import time

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up webcam
cap = cv2.VideoCapture(0)

# Get the initial time for frame rate calculation
start_time = time.time()

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display face coordinates
        cv2.putText(frame, f'X: {x}, Y: {y}, W: {w}, H: {h}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Crop the face region
        face_roi = frame[y:y + h, x:x + w]

        # Display the face region
        cv2.imshow('Face Region', face_roi)

    # Display the number of faces detected in the bottom-left corner
    cv2.putText(frame, f'Faces Detected: {len(faces)}', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

    # Display additional information in the top-left corner
    fps = 1 / (time.time() - start_time)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    frame_height, frame_width, _ = frame.shape

    cv2.putText(frame, f'Frame Rate: {fps:.2f} fps', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f'Timestamp: {current_time}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f'Frame Dimensions: {frame_width}x{frame_height}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Update the start time for the next iteration
    start_time = time.time()

    # Display the resulting frame with face detection
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()