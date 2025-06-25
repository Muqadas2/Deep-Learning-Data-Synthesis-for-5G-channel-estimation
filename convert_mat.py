from scipy.io import loadmat
import numpy as np
import os

# Set the directory path
# data_dir = r"C:\channel-est-dl\data\delayProfile\TDL-E"
data_dir = r"C:\channel-est-dl\data\snr\snr 8 10"

# Load files with full path
train_data_mat = loadmat(os.path.join(data_dir, "my_trainData.mat"))["trainData"]
train_labels_mat = loadmat(os.path.join(data_dir, "my_trainLabels.mat"))["trainLabels"]
val_data_mat = loadmat(os.path.join(data_dir, "my_valData.mat"))["valData"]
val_labels_mat = loadmat(os.path.join(data_dir, "my_valLabels.mat"))["valLabels"]
# test_data_mat = loadmat(os.path.join(data_dir, "my_testData.mat"))["testData"]
# test_labels_mat = loadmat(os.path.join(data_dir, "my_testLabels.mat"))["testLabels"]

print("Original Shapes:")
print("Train Data shape:", train_data_mat.shape)
print("Train Labels shape:", train_labels_mat.shape)
print("Validation Data shape:", val_data_mat.shape)
print("Validation Labels shape:", val_labels_mat.shape)
# print("Test Data shape:", test_data_mat.shape)
# print("Test Labels shape:", test_labels_mat.shape)

# Convert from (612, 14, 1, N) → (N, 612, 14, 1) for TensorFlow (NHWC)
train_data = np.transpose(train_data_mat, (3, 0, 1, 2))
train_labels = np.transpose(train_labels_mat, (3, 0, 1, 2))
val_data = np.transpose(val_data_mat, (3, 0, 1, 2))
val_labels = np.transpose(val_labels_mat, (3, 0, 1, 2))
# test_data = np.transpose(test_data_mat, (3, 0, 1, 2))
# test_labels = np.transpose(test_labels_mat, (3, 0, 1, 2))

print("Final TensorFlow-ready shapes:")
print("Train Data shape:", train_data.shape)
print("Train Labels shape:", train_labels.shape)
print("Validation Data shape:", val_data.shape)
print("Validation Labels shape:", val_labels.shape)
# print("Test Data shape:", test_data.shape)
# print("Test Labels shape:", test_labels.shape)

# Save as .npy files (in same folder)
np.save(os.path.join(data_dir, "tf_trainData.npy"), train_data)
np.save(os.path.join(data_dir, "tf_trainLabels.npy"), train_labels)
np.save(os.path.join(data_dir, "tf_valData.npy"), val_data)
np.save(os.path.join(data_dir, "tf_valLabels.npy"), val_labels)
# np.save(os.path.join(data_dir, "tf_testData.npy"), test_data)
# np.save(os.path.join(data_dir, "tf_testLabels.npy"), test_labels)

print("Converted to TensorFlow format and saved as .npy files.")
