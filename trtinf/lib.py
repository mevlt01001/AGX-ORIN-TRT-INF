import os
import tensorrt
import numpy
import pycuda.driver as cuda
import pycuda.autoinit, cv2
# import matplotlib.pyplot as plt
#import pyrealsense2 as rs
import time

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

        # cap = cv2.VideoCapture(0)
        # img = self.__get_img("test_imgs/273271-1ba93000e00e011c_jpg.rf.b86e6597a572425a278b60a8863e0227.jpg")
        # data = self.__preprocess(img)
        # self.__inference(data)
        # bboxes, classes = self.__postprocess(img)
        # self.__draw_boxes(img, bboxes, classes)
        
        self.__run_web_camera()
        # self.__run_realsense()

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
    
    def __get_img(self, cap=None):
        if not isinstance(cap, cv2.VideoCapture):
            return self.__get_img_from_file(cap)
        ret, img = cap.read()
        if not ret:
            print(f"{'[ERROR][TRT_INF:__get_img]':.<40}: Failed to read image")
            exit(1)
        img = cv2.resize(img, (640, 640))
        return img

    def __get_img_from_file(self, img_path: str):
        if not os.path.exists(img_path):
            print(f"{'[ERROR][TRT_INF:__get_img_from_file]':.<40}: The path '{img_path}' does not exist")
            exit(1)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        return img

    def __preprocess(self, img: numpy.ndarray):
        img = numpy.transpose(img, (2, 0, 1))
        img = numpy.expand_dims(img, axis=0)
        # print(f"{'[INFO][TRT_INF:__preprocess]':.<40}: Preprocess done")
        return img
    
    def __inference(self, data: numpy.ndarray):
        numpy.copyto(self.__input_buffers[0].cpu_buffer, data.ravel())
        # print(f"{'[INFO][TRT_INF:__inference]':.<40}: Data coppied to input buffer: {self.__input_buffers[0].cpu_buffer.base.get_device_pointer()}")
        cuda.memcpy_htod_async(self.__input_buffers[0].gpu_buffer, self.__input_buffers[0].cpu_buffer, self.__stream)
        self.__context.execute_async_v3(self.__stream.handle)
        cuda.memcpy_dtoh_async(self.__output_buffers[0].cpu_buffer, self.__output_buffers[0].gpu_buffer, self.__stream)
        cuda.memset_d32(self.__output_buffers[0].gpu_buffer, 0, self.__output_buffers[0].size)
        # self.__stream.synchronize()
        # print(f"{'[INFO][TRT_INF:__inference]':.<40}: Inference done")
    
    def __postprocess(self, img: numpy.ndarray, width_ratio: float = 1.0, height_ratio: float = 1.0):
        output = numpy.trim_zeros(self.__output_buffers[0].cpu_buffer, "b").reshape(-1, 7)
        output = output[numpy.isin(output[:, 1], list(classes.keys()))]
        scores = output[:, 2]
        mask = scores > 0.4
        output = output[mask]
        # print(f"{'[INFO][TRT_INF:__postprocess]':.<40}: Output shape: {output.shape}")
        
        bboxes = output[:, 3:]
        bboxes[:, 0] = bboxes[:, 0] * width_ratio
        bboxes[:, 1] = bboxes[:, 1] * height_ratio
        bboxes[:, 2] = bboxes[:, 2] * width_ratio
        bboxes[:, 3] = bboxes[:, 3] * height_ratio
        bboxes = bboxes.astype(int)
        _classes = output[:, 1].astype(int)
        return bboxes, _classes

    def __run_web_camera(self):
        cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg,format=MJPG ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            # print(f"{'[ERROR][TRT_INF:__run_web_camera]':.<40}: Failed to open camera")
            exit(1)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"{'[INFO][TRT_INF:__run_web_camera]':.<40}: Camera resolution: {w}x{h}")
        w_ratio = w / 640
        h_ratio = h / 640
        print(f"{'[INFO][TRT_INF:__run_web_camera]':.<40}: Width ratio: {w_ratio}, Height ratio: {h_ratio}")
        key = True
        cnt = 0
        fps = 0
        while key:
            start = time.time()
            key, img = cap.read()
            data = cv2.resize(img, (640, 640))
            data = self.__preprocess(data)
            self.__inference(data)
            bboxes, _classes = self.__postprocess(img, w_ratio, h_ratio)
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = box
                # print(f"{'[INFO][TRT_INF:__run_web_camera]':.<40}: {i}: {box}, {_classes[i]}")
                cv2.rectangle(img, (x1, y1), (x2, y2), classes[_classes[i]][1], 2)
                if _classes[i] in [3, 4]:
                    cv2.putText(img, classes[_classes[i]][0], (int(x1+(x2-x1)/2), int(y1-40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, classes[_classes[i]][1], 2)
                    continue
                cv2.putText(img, classes[_classes[i]][0], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, classes[_classes[i]][1], 2)
    
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Image", img)
            end = time.time()
            fps = 1 / (end - start)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cv2.imwrite(f"output_{cnt}.jpg", img)
                cnt += 1
                print(f"{'[INFO][TRT_INF:__run_web_camera]':.<40}: Image saved as output.jpg")
                # break
        cap.release()
        cv2.destroyAllWindows()
    

    def __run_realsense(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 640, rs.format.bgr8, 30)
        pipeline.start(config)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            img = numpy.asanyarray(color_frame.get_data())
            data = self.__preprocess(img)
            self.__inference(data)
            bboxes, classes = self.__postprocess(img)

            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, classes[classes[i]], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        pipeline.stop()
        cv2.destroyAllWindows()
    
    def __draw_boxes(self, img: numpy.ndarray, boxes: list, class_ids: list = None):
        print(f"{'[INFO][TRT_INF:__draw_boxes]':.<40}: num boxes: {len(boxes)}")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # print(f"{'[INFO][TRT_INF:__run_web_camera]':.<40}: {i}: {box}, {_classes[i]}")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if class_ids[i] in [3,4]:
                cv2.putText(img, classes[class_ids[i]], (int(x1+(x2-x1)/2), int(y1-40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                continue
            cv2.putText(img, classes[class_ids[i]], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imwrite("output.jpg", img)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    inf = TRT_INF("EngineFolder/yolov9_s_wholebody25_post_0100_1x3x640x640.engine")