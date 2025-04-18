import os
import tensorrt
import numpy
import pycuda.driver as cuda
import pycuda.autoinit

class Tensor_CPU_GPU_Buffers:
    def __init__(self, name, cpu_buffer, gpu_buffer):
        
        self.name = name
        self.cpu_buffer = cpu_buffer
        self.gpu_buffer = gpu_buffer

class TRT_INF:
    def __init__(self, engine_path: str):
        
        self.engine_path = engine_path
        self.__LOGGER, self.__RUNTIME, self.__ENGINE = self.__load_engine()
        self.__input_buffers, self.__output_buffers = self.__allocate_buffers()
        self.__context = self.__create_execution_context()
        self.__stream = cuda.Stream()
        
        for i in range(100):
            self.__inference(data=numpy.random.randn(1, 3, 640, 640).astype(numpy.float32))
            print(f"{'[INFO][TRT_INF:__inference]':.<40}: Inference {i+1} done")
        

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

    def __allocate_buffers(self):

        input_buffers = []
        output_buffers = []

        for tensor_idx in range(self.__ENGINE.num_io_tensors):
            tensor_name = self.__ENGINE.get_tensor_name(tensor_idx)
            tensor_dtype = tensorrt.nptype(self.__ENGINE.get_tensor_dtype(tensor_name))
            tensor_shape = self.__ENGINE.get_tensor_shape(tensor_name)
            tensor_mode = self.__ENGINE.get_tensor_mode(tensor_name)
            tensor_size = tensorrt.volume(tensor_shape)

            cpu_buffer = cuda.pagelocked_empty(tensor_size, tensor_dtype)
            gpu_buffer = cuda.mem_alloc(cpu_buffer.nbytes)

            buffer = Tensor_CPU_GPU_Buffers(tensor_name, cpu_buffer, gpu_buffer)

            info = f"""\n{'[INFO][TRT_INF:__allocate_buffers]':.<40}: 
            Tensor name: {tensor_name}
            Tensor shape: {tensor_shape}
            Tensor dtype: {tensor_dtype}
            Tensor size: {tensor_size}
            Tensor mode: {tensor_mode}
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
        for buffer in self.__input_buffers:
            context.set_tensor_address(buffer.name, int(buffer.gpu_buffer))
            print(f"{'[INFO][TRT_INF:__create_execution_context]':.<40}: {buffer.name} binded.")
        for buffer in self.__output_buffers:
            context.set_tensor_address(buffer.name, int(buffer.gpu_buffer))
            print(f"{'[INFO][TRT_INF:__create_execution_context]':.<40}: {buffer.name} binded.")
        return context
    
    def __inference(self, data: numpy.ndarray):
        cuda.memcpy_htod_async(self.__input_buffers[0].gpu_buffer, data, self.__stream)
        self.__context.execute_async_v3(self.__stream.handle)
        cuda.memcpy_dtoh_async(self.__output_buffers[0].cpu_buffer, self.__output_buffers[0].gpu_buffer, self.__stream)
        print(f"{'[INFO][TRT_INF:__inference]':.<40}: Inference done")
        
if __name__ == "__main__":
    inf = TRT_INF("yolov10l.engine")