import os
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from datetime import datetime
import onnx

# ==== CONFIG ====
engine_dir = "/home/embedaiot/trt_int8_engines/"
onnx_dir = "/home/embedaiot/Deep-Learning-Data-Synthesis-for-5G-channel-estimation/onnx_exports/"
val_data_path = "/home/embedaiot/dataset/tf_valData.npy"
val_labels_path = "/home/embedaiot/dataset/tf_valLabels.npy"
results_dir = "/home/embedaiot/int8_results/"
csv_path = os.path.join(results_dir, "int8_eval_results.csv")
batch_size = 8
os.makedirs(results_dir, exist_ok=True)

# ==== LOAD DATA ====
X_val = np.load(val_data_path).astype(np.float32)
y_val = np.load(val_labels_path).astype(np.float32)
num_samples = X_val.shape[0]

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ==== FUNCTIONS ====
def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, batch_shape):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        shape = engine.get_binding_shape(binding)
        shape = tuple(batch_shape if i == -1 else i for i in shape)
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_input):
    batch_input = batch_input.astype(inputs[0][0].dtype)
    context.set_binding_shape(0, batch_input.shape)
    np.copyto(inputs[0][0], batch_input.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()
    return outputs[0][0].reshape((batch_input.shape[0], -1))

def count_parameters_onnx(onnx_path):
    model = onnx.load(onnx_path)
    return sum(np.prod(t.dims) for t in model.graph.initializer)

# ==== WRITE CSV HEADER IF NOT EXISTS ====
file_exists = os.path.isfile(csv_path)
with open(csv_path, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "Timestamp", "Model", "Average MSE", "NMSE", "NMSE (dB)", "Total Samples",
            "Batch Inference Time (s)", "Avg Time per Sample (s)",
            "Single Sample Inference Time (s)", "Histogram Path",
            "Parameters", "Size (MB)", "FLOPs"
        ])

    # ==== LOOP OVER MODELS ====
    for engine_file in os.listdir(engine_dir):
        if not engine_file.endswith(".engine"):
            continue

        model_name = engine_file.replace("_INT8.engine", "")
        engine_path = os.path.join(engine_dir, engine_file)
        print(f"\n🔍 Evaluating: {model_name}")

        engine = load_engine(engine_path)
        context = engine.create_execution_context()
        batch_shape = (batch_size,) + X_val.shape[1:]
        inputs, outputs, bindings, stream = allocate_buffers(engine, batch_shape)

        preds = []
        total_batch_time = 0

        for i in range(0, len(X_val), batch_size):
            batch = X_val[i:i+batch_size]
            if len(batch) < batch_size:
                continue

            start = time.time()
            output = do_inference(context, bindings, inputs, outputs, stream, batch)
            end = time.time()
            preds.append(output)
            total_batch_time += (end - start)

        preds = np.vstack(preds)
        y_true = y_val[:preds.shape[0]].reshape(preds.shape)

        mse_per_sample = np.mean((preds - y_true) ** 2, axis=1)
        avg_mse = float(np.mean(mse_per_sample))

        nmse_numerator = np.sum((preds - y_true) ** 2)
        nmse_denominator = np.sum(y_true ** 2)
        nmse = nmse_numerator / nmse_denominator
        nmse_db = 10 * np.log10(nmse)

        avg_time_per_sample = total_batch_time / preds.shape[0]

        # Single sample time
        single_start = time.time()
        _ = do_inference(context, bindings, inputs, outputs, stream, X_val[0:1])
        single_end = time.time()
        single_time = single_end - single_start

        # Histogram
        hist_path = os.path.join(results_dir, f"mse_histogram_{model_name}.png")
        plt.figure(figsize=(8, 5))
        plt.hist(mse_per_sample, bins=10, color='skyblue', edgecolor='black')
        plt.title(f"MSE Histogram - {model_name}")
        plt.xlabel("MSE")
        plt.ylabel("Samples")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        # Params & FLOPs
        onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
        param_count = count_parameters_onnx(onnx_path)
        flops = 2 * param_count
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)

        print(f"✅ {model_name} - MSE: {avg_mse:.6f}, NMSE (dB): {nmse_db:.2f}, Time/sample: {avg_time_per_sample:.6f}s")

        # CSV row
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp, model_name,
            round(avg_mse, 8), round(nmse, 8), round(nmse_db, 4),
            len(y_true),
            round(total_batch_time, 6),
            round(avg_time_per_sample, 6),
            round(single_time, 6),
            os.path.abspath(hist_path),
            param_count, round(size_mb, 2), flops
        ])

print("\n✅ All INT8 models evaluated. Results saved to CSV.")
