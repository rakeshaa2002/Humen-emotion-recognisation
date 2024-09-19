from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os

# Load face classifier and emotion detection model
face_classifier = cv2.CascadeClassifier(r'D:\Sir_project\Emotion_Detection_CNN\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\Sir_project\Emotion_Detection_CNN\model.h5')

# Emotion labels and corresponding video paths
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
video_paths = {
    'Angry': r'D:\Sir_project\Emotion_Detection_CNN\videos\angry.mp4',
    'Disgust': r'D:\Sir_project\Emotion_Detection_CNN\videos\disgust.mp4',
    'Fear': r'D:\Sir_project\Emotion_Detection_CNN\videos\fear.mp4',
    'Happy': r'D:\Sir_project\Emotion_Detection_CNN\videos\happy.mp4',
    'Neutral': r'D:\Sir_project\Emotion_Detection_CNN\videos\neutral.mp4',
    'Sad': r'D:\Sir_project\Emotion_Detection_CNN\videos\sad.mp4',
    'Surprise': r'D:\Sir_project\Emotion_Detection_CNN\videos\surprise.mp4'
}

# Function to play video and return the current frame
def get_video_frame(video_cap):
    ret, frame = video_cap.read()
    if not ret:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
        ret, frame = video_cap.read()
    return frame

# Start video capture for emotion detection
cap = cv2.VideoCapture(0)

# Initialize video capture objects for each emotion
video_caps = {emotion: cv2.VideoCapture(path) for emotion, path in video_paths.items()}

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        emotion_label = "No Faces"
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                emotion_label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Create an output frame
        height, width, _ = frame.shape
        output_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)

        # Place the emotion detection frame on the left side
        output_frame[:, :width, :] = frame

        # Get the video frame for the detected emotion
        if emotion_label in video_caps:
            video_frame = get_video_frame(video_caps[emotion_label])
            video_frame = cv2.resize(video_frame, (width, height))
            output_frame[:, width:, :] = video_frame

        cv2.imshow('Emotion Detector and Video Recommendation', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release all resources
    cap.release()
    for vc in video_caps.values():
        vc.release()
    cv2.destroyAllWindows()
