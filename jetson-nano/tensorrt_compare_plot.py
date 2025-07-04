import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import onnx

# === Configuration ===
engine_dir = "/home/embedaiot/trt_engines/"
onnx_dir = "/home/embedaiot/Deep-Learning-Data-Synthesis-for-5G-channel-estimation/onnx_exports/"
data_dir = "/home/embedaiot/data_files/"
input_name = "tf_valData.npy"
label_name = "tf_valLabels.npy"
results_dir = "fp16_eval_results"
os.makedirs(results_dir, exist_ok=True)
output_csv = os.path.join(results_dir, "eval_results_fp16.csv")

# === Load and transpose data from NHWC → NCHW ===
val_data = np.load(Path(data_dir) / input_name).astype(np.float32)
val_labels = np.load(Path(data_dir) / label_name).astype(np.float32)

x_val = np.transpose(val_data, (0, 3, 1, 2))   # (N, C, H, W)
y_val = np.transpose(val_labels, (0, 3, 1, 2)) # (N, C, H, W)

# === TensorRT logger ===
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# === Helper to allocate buffers ===
def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

# === Count parameters from ONNX ===
def count_parameters(onnx_path):
    model = onnx.load(onnx_path)
    return sum(np.prod(t.dims) for t in model.graph.initializer)

# === Evaluation ===
with open(output_csv, "w") as f:
    f.write("Timestamp,Model,Average MSE,NMSE,NMSE (dB),Total Samples,"
            "Batch Inference Time (s),Avg Time per Sample (s),"
            "Single Sample Inference Time (s),Histogram Path\n")

    for engine_path in Path(engine_dir).glob("*.engine"):
        model_name = engine_path.stem
        onnx_path = Path(onnx_dir) / f"{model_name.split('_fp16')[0]}.onnx"

        with open(engine_path, "rb") as f_engine, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f_engine.read())
            context = engine.create_execution_context()

        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)

        preds, times = [], []
        for i in range(len(x_val)):
            np.copyto(h_input, x_val[i].ravel())
            start = time.time()
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            end = time.time()
            times.append(end - start)
            preds.append(h_output.copy())

        preds = np.array(preds).reshape(y_val.shape)
        mse_per_sample = np.mean((preds - y_val) ** 2, axis=(1, 2, 3))
        avg_mse = np.mean(mse_per_sample)
        nmse = np.sum((preds - y_val) ** 2) / np.sum(y_val ** 2)
        nmse_db = 10 * np.log10(nmse)

        total_time = np.sum(times)
        avg_time_per_sample = total_time / len(x_val)

        single_start = time.time()
        _ = context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        stream.synchronize()
        single_end = time.time()
        single_sample_time = single_end - single_start

        param_count = count_parameters(onnx_path)

        hist_path = os.path.join(results_dir, f"mse_histogram_{model_name}.png")
        plt.figure(figsize=(8, 5))
        plt.hist(mse_per_sample, bins=10, color='skyblue', edgecolor='black')
        plt.title(f"MSE Histogram - {model_name}")
        plt.xlabel("MSE")
        plt.ylabel("Samples")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp},{model_name},{avg_mse:.8f},{nmse:.8f},{nmse_db:.4f},"
                f"{len(x_val)},{total_time:.6f},{avg_time_per_sample:.6f},"
                f"{single_sample_time:.6f},{os.path.abspath(hist_path)}\n")

print(f"\n✅ Evaluation complete. All metrics saved to {output_csv}")
