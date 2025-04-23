import tensorrt, os
from sog4onnx import generate
MODEL_PATH = "resources_s/resources_e_withpost/yolov9_e_wholebody28_refine_post_0100_1x3x544x1280.onnx"
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
        for i in range(PARSER.num_errors):
            print(PARSER.get_error(i))
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
print(f"{'[SUCCESS][ENGINE]':.<40}: The engine is saved to {ENGINE_PATH}")