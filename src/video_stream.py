import cv2
import time

class VideoStream:
    def __init__(self,camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.prev_time = 0

        if not self.cap.isOpened():
            raise ValueError("Camera not accessible")
    
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Error: Failed to grab frame. retrying...")
            time.sleep(0.05)
            return self.read_frame()
        return frame
        
    
    def calculate_fps(self):
        current_time = time.time()
        fps = 1/(current_time-self.prev_time) if self.prev_time else 0
        self.prev_time = current_time
        return round(fps,2)
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()



