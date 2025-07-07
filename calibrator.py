# calibrator.py
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data_path, batch_size=50):
        super(MyCalibrator, self).__init__()
        self.batch_size = batch_size
        self.data = np.load(calibration_data_path)
        self.data = self.data.astype(np.float32)
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.data):
            return None
        batch = self.data[self.current_index:self.current_index + self.batch_size]
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        with open("/home/embedaiot/calibration.cache", "wb") as f:
            f.write(cache)

