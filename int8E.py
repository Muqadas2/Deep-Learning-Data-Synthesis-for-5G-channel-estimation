# build engine int8 
import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from calibrator import MyCalibrator

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_int8_engine(onnx_model_path, engine_file_path, calibration_data_path):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_model_path, "rb") as f:
        if not parser.parse(f.read()):
            print(f"[] Failed to parse ONNX: {onnx_model_path}")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # INT8 flags
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    # Load calibrator
    calibrator = MyCalibrator(calibration_data_path, batch_size=50)
    config.int8_calibrator = calibrator

    # Dynamic input shape profile
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape  # e.g., [None, 612, 14, 1]

    fallback_shape = [dim if isinstance(dim, int) else 612 if i == 1 else 14 if i == 2 else 1
                      for i, dim in enumerate(input_shape)]

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(1, *fallback_shape[1:]),
        opt=(8, *fallback_shape[1:]),
        max=(16, *fallback_shape[1:])
    )
    config.add_optimization_profile(profile)

    # Build engine
    print(f"Building INT8 engine for: {onnx_model_path}")
    engine = builder.build_engine(network, config)

    if engine:
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print(f" Saved TensorRT engine to: {engine_file_path}")
        return engine
    else:
        print(f" Engine build failed: {onnx_model_path}")
        return None

if __name__ == "__main__":
    onnx_dir = "/home/embedaiot/onnx_exports_old/"
    engine_dir = "/home/embedaiot/trt_int8_engines_old"
    calibration_path = "/home/embedaiot/calib_data.npy"

    os.makedirs(engine_dir, exist_ok=True)

    for fname in os.listdir(onnx_dir):
        if fname.endswith(".onnx"):
            model_path = os.path.join(onnx_dir, fname)
            engine_name = fname.replace(".onnx", "_INT8.engine")
            engine_path = os.path.join(engine_dir, engine_name)

            build_int8_engine(model_path, engine_path, calibration_path)

