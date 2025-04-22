import tensorrt, os


onnx_files = {
    "resources_s/yolov9_s_wholebody25_post_0100_1x3x544x960.onnx": (1, 3, 544, 960),
    "resources_s/yolov9_s_wholebody25_post_0100_1x3x544x1280.onnx": (1, 3, 544, 1280),
    "resources_s/yolov9_s_wholebody25_post_0100_1x3x576x1024.onnx": (1, 3, 576, 1024),
    "resources_s/yolov9_s_wholebody25_post_0100_1x3x640x640.onnx": (1, 3, 640, 640),
    "resources_s/yolov9_s_wholebody25_post_0100_1x3x736x1280.onnx": (1, 3, 736, 1280),
}

MODEL_PATH = "resources_s/yolov9_s_wholebody25_post_0100_1x3x736x1280.onnx"
os.makedirs("EngineFolder", exist_ok=True)
MODEL_NAME = MODEL_PATH.lstrip("resources_s/")
MODEL_NAME = MODEL_NAME.replace(".onnx", ".engine")
ENGINE_PATH = os.path.join("EngineFolder", MODEL_NAME)
# if os.path.exists(ENGINE_PATH):
#     print(f"{'[INFO][ENGINE]':.<40}: The engine already exists")
#     exit(0)
LOGGER = tensorrt.Logger(tensorrt.Logger.VERBOSE)
BUILDER = tensorrt.Builder(LOGGER)
NETWORK = BUILDER.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
PARSER = tensorrt.OnnxParser(NETWORK, LOGGER)
# Load the ONNX model
with open(MODEL_PATH, "rb") as model:
    if not PARSER.parse(model.read()):
        print(f"{'[ERROR][ONNX_PARSER]':.<40}: {PARSER.get_error(0)}")
        exit(1)

PROFILE = BUILDER.create_optimization_profile()
PROFILE.set_shape(NETWORK.get_input(0).name, (1, 3, 720, 1280), (1, 3, 720, 1280), (1, 3, 720, 1280))

CONFIG = BUILDER.create_builder_config()
CONFIG.profiling_verbosity = tensorrt.ProfilingVerbosity.DETAILED
CONFIG.set_memory_pool_limit(tensorrt.MemoryPoolType.WORKSPACE, 16*1024*1024*1024)  # 16GB
CONFIG.add_optimization_profile(PROFILE)

if BUILDER.platform_has_fast_fp16:
    CONFIG.set_flag(tensorrt.BuilderFlag.FP16)

ENGINE = BUILDER.build_serialized_network(NETWORK, CONFIG)
if ENGINE is None:
    print(f"{'[ERROR][ENGINE]':.<40}: The engine is empty")
    exit(1)
print(f"{'[SUCCESS][ENGINE]':.<40}: The engine is created")
# Save the engine to a file
with open(ENGINE_PATH, "wb") as f:
    f.write(ENGINE)