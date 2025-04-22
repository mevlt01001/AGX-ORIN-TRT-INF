import os
import tensorrt
import numpy
import pycuda.driver as cuda
import pycuda.autoinit, cv2
# import matplotlib.pyplot as plt
#import pyrealsense2 as rs
import time
import threading
from collections import deque

class CameraBuffer:
    def __init__(self, maxsize=20):
        self.cap = cv2.VideoCapture(4)
        if not self.cap.isOpened():
            raise RuntimeError("Kamera açılamadı.")
        self.queue = deque(maxlen=maxsize)
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=False)

    def _reader(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.queue.append(frame)
                    print(f"{'[INFO][CameraBuffer:_reader]':.<40}: queeue size: {len(self.queue)}")
            
    def get(self):
        with self.lock:
            if len(self.queue) > 0:
                return self.queue.pop()
            else:
                return None


classes = {
    # 0: 'Body',
    # 1: 'Adult',
    # 2: 'Child',
    3: ('Body(M)', (0, 0, 255)),
    4: ('Body(F)', (255, 0, 255)),
    # 5: 'Body_with_Wheelchair',
    # 6: 'Body_with_Crutches',
    # 7: 'Head',
    8: ('Head(F)', (0, 255, 0)),
    9: ('Head(RF)', (0, 255, 0)),
    10: ('Head(R)', (0, 255, 0)),
    11: ('Head(RB)', (0, 255, 0)),
    12: ('Head(B)', (0, 255, 0)),
    13: ('Head(LB)', (0, 255, 0)),
    14: ('Head(L)', (0, 255, 0)),
    15: ('Head(LF)', (0, 255, 0)),
    16: ('', (120,120,120)),#Face
    17: ('', (20,120,240)),#Eye
    18: ('',(190, 200,70)),#Nose
    19: ('', (250, 20,120)),#Mouth
    20: ('', (200,255,170)),#Ear
    # 21: 'Hand',
    22: ('Hand(L)', (0, 255, 0)),
    23: ('Hand(R)', (0, 255, 0)),
    24: ('Foot', (0, 255, 0)),
}

class Tensor_CPU_GPU_Buffers:
    def __init__(self, name, cpu_buffer, gpu_buffer, size, dtype, mode, shape):
        
        self.name = name
        self.shape = shape
        self.size = size
        self.dtype = dtype
        self.mode = mode
        self.cpu_buffer = cpu_buffer
        self.gpu_buffer = gpu_buffer

class TRT_INF:
    def __init__(self, engine_path: str):
        
        self.engine_path = engine_path
        self.__LOGGER, self.__RUNTIME, self.__ENGINE = self.__load_engine()
        self.__input_buffers, self.__output_buffers = self.__allocate_buffers(outShape=(1000,7))
        self.__context, self.__stream = self.__create_execution_context()

    def __load_engine(self):
        if not os.path.exists(self.engine_path): print(f"{'[ERROR][TRT_INF:__load_engine]':.<40}:The path '{self.engine_path}' does not exist"), exit(1)
        
        LOGGER = tensorrt.Logger(tensorrt.Logger.ERROR)
        RUNTIME = tensorrt.Runtime(LOGGER)
        engine_file = open(self.engine_path, "rb")
        engine = RUNTIME.deserialize_cuda_engine(engine_file.read())
        engine_file.close()
        if engine is None: print(f"{'[ERROR][TRT_INF:__load_engine]':.<40}:The engine is empty"), exit(1)
        print(f"{'[SUCCESS][TRT_INF:__load_engine]':.<40}: {self.engine_path} loaded")
        return LOGGER, RUNTIME, engine

    def __allocate_buffers(self, outShape=None):

        input_buffers = []
        output_buffers = []

        for tensor_idx in range(self.__ENGINE.num_io_tensors):
            tensor_name = self.__ENGINE.get_tensor_name(tensor_idx)
            tensor_dtype = tensorrt.nptype(self.__ENGINE.get_tensor_dtype(tensor_name))
            tensor_shape = self.__ENGINE.get_tensor_shape(tensor_name) if not -1 in self.__ENGINE.get_tensor_shape(tensor_name) else outShape
            tensor_mode = self.__ENGINE.get_tensor_mode(tensor_name)
            tensor_size = tensorrt.volume(tensor_shape)

            info = f"""\n{'[INFO][TRT_INF:__allocate_buffers]':.<40}: 
            Tensor name: {tensor_name}
            Tensor shape: {tensor_shape}
            Tensor dtype: {tensor_dtype}
            Tensor size: {tensor_size}
            Tensor mode: {tensor_mode}"""
            print(info, end="")

            cpu_buffer = cuda.pagelocked_empty(tensor_size, tensor_dtype)
            gpu_buffer = cuda.mem_alloc(cpu_buffer.nbytes)

            buffer = Tensor_CPU_GPU_Buffers(tensor_name, cpu_buffer, gpu_buffer, tensor_size, tensor_dtype, tensor_mode, tensor_shape)


            info = f"""
            Shared memory at GPU: {int(buffer.gpu_buffer)}, {buffer.cpu_buffer.nbytes} bytes
            Shared memory at CPU: {int(buffer.cpu_buffer.base.get_device_pointer())}, {buffer.cpu_buffer.nbytes} bytes
            """
            print(info)

            if tensor_mode == tensorrt.TensorIOMode.INPUT:
                input_buffers.append(buffer)
            if tensor_mode == tensorrt.TensorIOMode.OUTPUT:
                output_buffers.append(buffer)
        
        return input_buffers, output_buffers

    def __create_execution_context(self):
        context = self.__ENGINE.create_execution_context()
        print(f"{'[INFO][TRT_INF:__create_execution_context]':.<40}: Context created")  
        for buffer in self.__input_buffers:
            context.set_tensor_address(buffer.name, int(buffer.gpu_buffer))
            print(f"{'[INFO][TRT_INF:__create_execution_context]':.<40}: {buffer.name} binded.")
        for buffer in self.__output_buffers:
            context.set_tensor_address(buffer.name, int(buffer.gpu_buffer))
            print(f"{'[INFO][TRT_INF:__create_execution_context]':.<40}: {buffer.name} binded.")
        return context, cuda.Stream()

    def __preprocess(self, img: numpy.ndarray):
        # shape is (H, W, 3)
        img = numpy.transpose(img, (2, 0, 1))
        img = numpy.expand_dims(img, axis=0)
        return img
    
    def __inference(self, data: numpy.ndarray):
        numpy.copyto(self.__input_buffers[0].cpu_buffer, data.ravel())
        cuda.memcpy_htod_async(self.__input_buffers[0].gpu_buffer, self.__input_buffers[0].cpu_buffer, self.__stream)
        self.__context.execute_async_v3(self.__stream.handle)
        cuda.memcpy_dtoh_async(self.__output_buffers[0].cpu_buffer, self.__output_buffers[0].gpu_buffer, self.__stream)
        cuda.memset_d32(self.__output_buffers[0].gpu_buffer, 0, self.__output_buffers[0].size)
        self.__stream.synchronize()
    
    def __postprocess(self, img: numpy.ndarray, width_ratio: float = 1.0, height_ratio: float = 1.0):
        output = numpy.trim_zeros(self.__output_buffers[0].cpu_buffer, "b").reshape(-1, 7)
        output = output[numpy.isin(output[:, 1], list(classes.keys()))]
        scores = output[:, 2]
        mask = scores > 0.5
        output = output[mask]
        
        bboxes = output[:, 3:]
        bboxes[:, 0] = bboxes[:, 0] * width_ratio
        bboxes[:, 1] = bboxes[:, 1] * height_ratio
        bboxes[:, 2] = bboxes[:, 2] * width_ratio
        bboxes[:, 3] = bboxes[:, 3] * height_ratio
        bboxes = bboxes.astype(int)
        _classes = output[:, 1].astype(int)
        return bboxes, _classes

    def run_web_camera(self):
        scale_f = 1
        w = 1280*scale_f
        h = 720*scale_f
        w_ratio, h_ratio = w / 1280, h / 736

        camera = CameraBuffer()
        camera.cap.set(cv2.CAP_PROP_FPS, 30)
        camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w/scale_f)
        camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h/scale_f)
        camera.thread.start()

        print(f"{'[INFO][TRT_INF:__run_web_camera]':.<40}: Camera resolution: {w}x{h}")
        print(f"{'[INFO][TRT_INF:__run_web_camera]':.<40}: Width ratio: {w_ratio}, Height ratio: {h_ratio}")

        fpss = deque(maxlen=50)
        pre_op_latency = deque(maxlen=50)
        inf_op_latency = deque(maxlen=50)
        post_op_latency = deque(maxlen=50)
        fps = 1
        pre_latency = 1
        inf_latency = 1
        post_latency = 1
        while True:
            start = time.perf_counter()

            fpss.append(fps)
            pre_op_latency.append(pre_latency)
            inf_op_latency.append(inf_latency)
            post_op_latency.append(post_latency)

            pre_start = time.perf_counter()
            img = camera.get()
            if img is None:
                continue
            # img2 = cv2.resize(img, (w, h))
            data = self.__preprocess(img)
            pre_end = time.perf_counter()

            inf_start = time.perf_counter()
            self.__inference(data)
            inf_end = time.perf_counter()

            post_start = time.perf_counter()
            bboxes, _classes = self.__postprocess(img, w_ratio, h_ratio)
            cv2.putText(img, f"FPS: {int(sum(fpss)/len(fpss))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"PreOPLat: {(sum(pre_op_latency)/len(pre_op_latency)):.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"InfOPLat: {(sum(inf_op_latency)/len(inf_op_latency)):.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"PostOPLat: {(sum(post_op_latency)/len(post_op_latency)):.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = box

                cv2.rectangle(img, (x1, y1), (x2, y2), classes[_classes[i]][1], 2)
                if _classes[i] in [3, 4]:
                    cv2.putText(img, classes[_classes[i]][0], (int(x1+(x2-x1)/2), int(y1-40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, classes[_classes[i]][1], 2)
                    continue
                cv2.putText(img, classes[_classes[i]][0], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, classes[_classes[i]][1], 2)
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            pre_latency = (pre_end - pre_start)*1000
            inf_latency = (inf_end - inf_start)*1000
            post_end = time.perf_counter()
            post_latency = (post_end - post_start)*1000
            fps = 1 / (time.perf_counter() - start)
            
        camera.running = False
        camera.thread.join()
        camera.cap.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    inf = TRT_INF("EngineFolder/yolov9_s_wholebody25_post_0100_1x3x736x1280.engine")
    inf.run_web_camera()

