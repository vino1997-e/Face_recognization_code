# Face_recognization_code
pip install opencv-python numpy face-recognition

import face_recognition
import cv2
import numpy as np

# Step 1: Load and Encode the Uploaded Image
uploaded_image_path = "C:/Users/Vinoth/OneDrive - Quation Solutions Private Limited/Desktop/vino.png"  # Replace with your actual image file
known_image = face_recognition.load_image_file(uploaded_image_path)
known_face_encodings = face_recognition.face_encodings(known_image)

if not known_face_encodings:
    print("Error: No faces found in the uploaded image.")
    exit()

known_face_encoding = known_face_encodings[0]  # Assume one face in the image

# Step 2: Capture Webcam Image and Recognize Face
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Could not capture an image from the webcam")
        break

    # Convert frame to RGB (face_recognition works with RGB format, not BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized = False

    # Loop over each face found in the current frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare faces
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        
        # Check if the face matches
        if True in matches:
            recognized = True
            top, right, bottom, left = face_location
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle
            cv2.putText(frame, "Face Recognized!", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        else:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Red rectangle
            cv2.putText(frame, "No Match Found", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show the current frame with recognized faces
    cv2.imshow('Webcam Face Recognition', frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
