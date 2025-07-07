import numpy as np

data = np.load("/home/embedaiot/dataset/tf_trainData.npy", allow_pickle=True)
data = data.astype(np.float32)

# Save a few samples (e.g., 100)
np.save("/home/embedaiot/calib_data.npy", data[:100])

