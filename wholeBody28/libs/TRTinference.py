import tensorrt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

class buffer:
    def __init__(self, name:str, dummy:np.ndarray):
        self.name = name
        self.dtype = dummy.dtype
        self.shape = dummy.shape
        self.size = dummy.size
        self.nbytes = dummy.nbytes
        self.device = cuda.mem_alloc_like(dummy)
        self.host = cuda.pagelocked_empty_like(dummy)
        
    def __repr__(self):
        return f"Buffer(name={self.name}, dtype={self.dtype}, shape={self.shape})"

class TRTInfer:
    def __init__(self, engine_path:os.PathLike, input_shape:tuple=None, output_shape:tuple=None):
        self.engine_path = engine_path
        self.LOGGER = tensorrt.Logger(tensorrt.Logger.VERBOSE)
        self.BUILDER = tensorrt.Builder(self.LOGGER)
        self.CONFIG = self.BUILDER.create_builder_config()
        self.NETWORK = self.BUILDER.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.PARSER = tensorrt.OnnxParser(self.NETWORK, self.LOGGER)
        self.ENGINE = None
    
    def create_INetworkDefinition(self, onnx_path:os.PathLike, input_shape:tuple):
        assert os.path.exists(onnx_path), f"ONNX file {onnx_path} does not exist"
        with open(onnx_path, "rb") as model:
            rotobuf = model.read()
            if not self.PARSER.parse(rotobuf):
                for error in range(self.PARSER.num_errors):
                    print(f"{'[ERROR][ONNX_PARSER]':.<40}: {self.PARSER.get_error(error)}")
                exit(1)
        print(f"{'[SUCCESS][ONNX_PARSER]':.<40}: The ONNX model is parsed")
        self.PROFILE = self.BUILDER.create_optimization_profile()
        self.PROFILE.set_shape(self.NETWORK.get_input(0).name, input_shape, input_shape, input_shape)

    def create_ICudaEngine(self, onnx_path:os.PathLike, input_shape:tuple):
        self.create_INetworkDefinition(onnx_path, input_shape)
        self.CONFIG.profiling_verbosity = tensorrt.ProfilingVerbosity.DETAILED
        self.CONFIG.set_memory_pool_limit(tensorrt.MemoryPoolType.WORKSPACE, 8*1024*1024*1024)
        self.CONFIG.add_optimization_profile(self.PROFILE)
        if self.BUILDER.platform_has_fast_fp16:
            self.CONFIG.set_flag(tensorrt.BuilderFlag.FP16)
        self.ENGINE = self.BUILDER.build_serialized_network(self.NETWORK, self.CONFIG)
        if self.ENGINE is None:
            print(f"{'[ERROR][ENGINE]':.<40}: The engine is empty")
            exit(1)
        print(f"{'[SUCCESS][ENGINE]':.<40}: The engine is created")
        with open(self.engine_path, "wb") as f:
            f.write(self.ENGINE)
        print(f"{'[SUCCESS][ENGINE]':.<40}: The engine is saved to {self.engine_path}")
        
    def load_engine(self, output_shape:tuple=None):
        assert os.path.exists(self.engine_path), f"Engine file {self.engine_path} does not exist"
        with open(self.engine_path, "rb") as f:
            ENGINE = f.read()
        self.RUNTIME = tensorrt.Runtime(self.LOGGER)
        self.ENGINE = self.RUNTIME.deserialize_cuda_engine(ENGINE)
        if self.ENGINE is None:
            print(f"{'[ERROR][ENGINE]':.<40}: The engine is empty")
            exit(1)
        print(f"{'[SUCCESS][ENGINE]':.<40}: The engine is loaded from {self.engine_path}")
        self.allocate_buffers(output_shape)
        self.create_context()
        print(f"{'[SUCCESS][ENGINE]':.<40}: The engine is created")

    def allocate_buffers(self, output_shape:tuple=None):
        self.input_buffers = []
        self.output_buffers = []

        for i in range(self.ENGINE.num_io_tensors):
            name = self.ENGINE.get_tensor_name(i)
            shape = self.ENGINE.get_tensor_shape(name) if self.ENGINE.get_tensor_mode(name) == tensorrt.TensorIOMode.INPUT else output_shape
            dtype = self.ENGINE.get_tensor_dtype(name)
            dtype = tensorrt.nptype(dtype)
            mode = self.ENGINE.get_tensor_mode(name)
            dummy = np.zeros(shape, dtype=dtype)
            
            info = f"""
                    name: {name}:
                    shape: {shape}
                    dtype: {dtype}
                    mode: {mode}
            """
            print(f"{'[INFO][BUFFER]':.<40}: {info}")

            _buffer = buffer(name, dummy)

            if mode == tensorrt.TensorIOMode.OUTPUT:
                self.output_buffers.append(_buffer)
            elif mode == tensorrt.TensorIOMode.INPUT:
                self.input_buffers.append(_buffer)

        print(f"{'[SUCCESS][BUFFER]':.<40}: The buffers are allocated")

    def create_context(self):
        self.context = self.ENGINE.create_execution_context()
        if self.context is None:
            print(f"{'[ERROR][CONTEXT]':.<40}: The engine.context is empty")
            exit(1)
        print(f"{'[SUCCESS][CONTEXT]':.<40}: The engine.context is created")
        for buffer in self.input_buffers:
            self.context.set_tensor_address(buffer.name, buffer.device)
            print(f"{'[INFO][CONTEXT]':.<40}: input buffer {buffer.name}({buffer.shape}) bindend to Context")
        for buffer in self.output_buffers:
            self.context.set_tensor_address(buffer.name, buffer.device)
            print(f"{'[INFO][CONTEXT]':.<40}: output buffer {buffer.name}({buffer.shape}) bindend to Context")
        self.stream = cuda.Stream()
        print(f"{'[SUCCESS][CONTEXT]':.<40}: cuda.Stream is created")
    
    def bind(self, input_data:np.ndarray):

        cuda.memcpy_htod_async(self.input_buffers[0].device, self.input_buffers[0].host, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_buffers[0].host, self.output_buffers[0].device, self.stream)
        cuda.memset_d32_async(self.output_buffers[0].device, 0, self.output_buffers[0].size, self.stream)
        self.stream.synchronize()

        
    
        
