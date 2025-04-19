import os
import tensorrt
import numpy
import pycuda.driver as cuda
import pycuda.autoinit, cv2
import matplotlib.pyplot as plt

classes = {
    0: 'Body',
    1: 'Adult',
    2: 'Child',
    3: 'Male',
    4: 'Female',
    5: 'Body_with_Wheelchair',
    6: 'Body_with_Crutches',
    7: 'Head',
    8: 'Front',
    9: 'Right_Front',
    10: 'Right_Side',
    11: 'Right_Back',
    12: 'Back',
    13: 'Left_Back',
    14: 'Left_Side',
    15: 'Left_Front',
    16: 'Face',
    17: 'Eye',
    18: 'Nose',
    19: 'Mouth',
    20: 'Ear',
    21: 'Hand',
    22: 'Hand_Left',
    23: 'Hand_Right',
    24: 'Foot'  
}

class Tensor_CPU_GPU_Buffers:
    def __init__(self, name, cpu_buffer, gpu_buffer):
        
        self.name = name
        self.cpu_buffer = cpu_buffer
        self.gpu_buffer = gpu_buffer

class TRT_INF:
    def __init__(self, engine_path: str):
        
        self.engine_path = engine_path
        self.__LOGGER, self.__RUNTIME, self.__ENGINE = self.__load_engine()
        self.__input_buffers, self.__output_buffers = self.__allocate_buffers(outShape=(100,7))
        self.__context, self.__stream = self.__create_execution_context()
        cap = cv2.VideoCapture(0)
        img = self.__get_img("test_imgs/273271-2b427000e2a2b025_jpg.rf.7d933851f233dcd09cf166e310a4b407.jpg")
        print(f"{'[INFO][TRT_INF:__init__]':.<40}: IMG SHAPE: {img.shape}")
        data = self.__preprocess(img)
        self.__inference(data)
        bboxes, classes = self.__postprocess(img)
        self.__draw_boxes(img, bboxes, classes)
        

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

            buffer = Tensor_CPU_GPU_Buffers(tensor_name, cpu_buffer, gpu_buffer)


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
            return self.__get_img_from_file("test_imgs/273271-1ba93000e00e011c_jpg.rf.b86e6597a572425a278b60a8863e0227.jpg")
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
        print(f"{'[INFO][TRT_INF:__preprocess]':.<40}: Preprocess done")
        return numpy.ascontiguousarray(img, dtype=numpy.float32)
    
    def __inference(self, data: numpy.ndarray):
        numpy.copyto(self.__input_buffers[0].cpu_buffer, data.ravel())
        print(f"{'[INFO][TRT_INF:__inference]':.<40}: Data coppied to input buffer: {self.__input_buffers[0].cpu_buffer.base.get_device_pointer()}")
        cuda.memcpy_htod_async(self.__input_buffers[0].gpu_buffer, self.__input_buffers[0].cpu_buffer, self.__stream)
        self.__context.execute_async_v3(self.__stream.handle)
        cuda.memcpy_dtoh_async(self.__output_buffers[0].cpu_buffer, self.__output_buffers[0].gpu_buffer, self.__stream)
        # self.__stream.synchronize()
        print(f"{'[INFO][TRT_INF:__inference]':.<40}: Inference done")
    
    def __postprocess(self, img: numpy.ndarray):
        output = numpy.trim_zeros(self.__output_buffers[0].cpu_buffer, "b").reshape(-1, 7)
        print(f"{'[INFO][TRT_INF:__postprocess]':.<40}: Output shape: {output.shape}")
        bboxes = (output[..., 3:].astype(numpy.int32)).tolist()
        conf = output[..., 2].astype(numpy.float32).tolist()
        class_ids = (output[..., 1].astype(numpy.int32)).tolist()

        print(f"{'[INFO][TRT_INF:__postprocess]':.<40}: Postprocess done")
        return bboxes, class_ids
    
    def __get_box_and_class(self, boxes: list, class_ids: list, scores: list):
        pass
        
    def __run(self, img: numpy.ndarray):
        img = self.__preprocess(img)
        self.__inference(img)
        output = self.__postprocess(img)
        return output
    
    def __draw_boxes(self, img: numpy.ndarray, boxes: list, class_ids: list = None):
        print(f"{'[INFO][TRT_INF:__draw_boxes]':.<40}: num boxes: {len(boxes)}")
        for i,box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            if class_ids is not None:
                cv2.putText(img, str(class_ids[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 1)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inf = TRT_INF("EngineFolder/yolov9_s_wholebody25_post_0100_1x3x640x640.engine")