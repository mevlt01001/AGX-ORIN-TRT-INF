from collections import deque
from threading import Thread, Lock
import cv2

class CameraAsync:
    def __init__(self, camera_index=0, queue_size=100, pipeline=None):
        self.cam = cv2.VideoCapture(camera_index) if pipeline is None else cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        self.frame_queue = deque(maxlen=queue_size)
        self.is_running = True
        self.lock = Lock()
        self.thread = Thread(target=self._reader)

    def _reader(self):
        while self.is_running:
            success, frame = self.cam.read()
            if success:
                self.frame_queue.append(frame)
                print(f"{'[INFO][CAMERA]':.<40}: frame {len(self.frame_queue)} is captured")

    def get(self):
        if len(self.frame_queue) > 0:
            return self.frame_queue.pop()

    def start(self):
        self.is_running = True
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.thread.join()
        self.cam.release()
        cv2.destroyAllWindows()
            