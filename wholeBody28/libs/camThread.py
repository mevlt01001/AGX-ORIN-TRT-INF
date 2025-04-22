from collections import deque
from threading import Thread, Lock
import cv2

class CameraAsync:
    def __init__(self, camera_index=0, queue_size=20):
        self.cam = cv2.VideoCapture(camera_index)
        self.frame_queue = deque(maxlen=queue_size)
        self.is_running = True
        self.lock = Lock()
        self.thread = Thread(target=self._reader)

    def _reader(self):
        while self.is_running:
            success, frame = self.cam.read()
            if success:
                with self.lock:       
                    self.frame_queue.append(frame)

    def get(self):
        with self.lock:
            if len(self.frame_queue) > 0:
                return self.frame_queue.pop()
            else:
                return None
            