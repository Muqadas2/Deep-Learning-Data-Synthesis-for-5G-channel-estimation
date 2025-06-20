from scipy.io import loadmat
import numpy as np
import os

# Set the directory path
data_dir = r"C:\Users\emana\OneDrive - National University of Sciences & Technology\Documents\summer 2025\channel-est-dl\data"

# Load files with full path
# train_data_mat = loadmat(os.path.join(data_dir, "my_trainData.mat"))["trainData"]
# train_labels_mat = loadmat(os.path.join(data_dir, "my_trainLabels.mat"))["trainLabels"]
# val_data_mat = loadmat(os.path.join(data_dir, "my_valData.mat"))["valData"]
# val_labels_mat = loadmat(os.path.join(data_dir, "my_valLabels.mat"))["valLabels"]
# test data
test_data_mat = loadmat(os.path.join(data_dir, "my_testData.mat"))["testData"]
test_labels_mat = loadmat(os.path.join(data_dir, "my_testLabels.mat"))["testLabels"]


# print("Train Data shape:", train_data_mat.shape)
# print("Train Labels shape:", train_labels_mat.shape)
# print("Validation Data shape:", val_data_mat.shape)
# print("Validation Labels shape:", val_labels_mat.shape)
# test data and labels shape
print("Test Data shape:", test_data_mat.shape)
print("Test Labels shape:", test_labels_mat.shape)


# Convert from (612, 14, 1, N) → (N, 1, 612, 14)
# train_data = np.transpose(train_data_mat, (3, 2, 0, 1))
# train_labels = np.transpose(train_labels_mat, (3, 2, 0, 1))
# val_data = np.transpose(val_data_mat, (3, 2, 0, 1))
# val_labels = np.transpose(val_labels_mat, (3, 2, 0, 1))
test_data = np.transpose(test_data_mat, (3, 2, 0, 1))
test_labels = np.transpose(test_labels_mat, (3, 2, 0, 1))


print("Final PyTorch-ready shapes:")
# print("Train Data shape:", train_data.shape)
# print("Train Labels shape:", train_labels.shape)
# print("Validation Data shape:", val_data.shape)
# print("Validation Labels shape:", val_labels.shape)
print("Test Data shape:", test_data.shape)
print("Test Labels shape:", test_labels.shape)  


# Save as .npy files (in same folder)
# np.save(os.path.join(data_dir, "my_trainData.npy"), train_data)
# np.save(os.path.join(data_dir, "my_trainLabels.npy"), train_labels)
# np.save(os.path.join(data_dir, "my_valData.npy"), val_data)
# np.save(os.path.join(data_dir, "my_valLabels.npy"), val_labels)
np.save(os.path.join(data_dir, "my_testData.npy"), test_data)
np.save(os.path.join(data_dir, "my_testLabels.npy"), test_labels)

print("Converted and saved as .npy files")