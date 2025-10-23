from src.video_stream import VideoStream
from src.face_mesh_detector import FaceMeshDetector
from src.emotion_detector import EmotionDetector
import cv2
import os
import time

stream = VideoStream()
detector = FaceMeshDetector()
emotion_detector = EmotionDetector()

screenshot_dir = "screenshot"
os.makedirs(screenshot_dir, exist_ok=True)
screenshot_count = 0

while True:
    frame = stream.read_frame()
    fps = stream.calculate_fps()

    result = detector.process(frame)
    frame = detector.draw(frame, result)

# If face landmarks detected, analyze emotion
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            state = emotion_detector.predict_emotion(face_landmarks.landmark, frame.shape)
            # Display emotion state on frame
            cv2.putText(frame, f'Emotion: {state}', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    cv2.putText(frame,f'FPS: {fps}',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("webcam Stream: FaceMesh Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'): # Quit program
        break
    elif key == ord('s'): # Save screenshot
        timestamp = time.strftime("%d%m%Y-%H%M%S")
        file=os.path.join(screenshot_dir, f'screenshot_{timestamp}.png')
        cv2.imwrite(file, frame)
        screenshot_count += 1
        print(f'Screenshot saved: {file} (Total: {screenshot_count})')

stream.release()