# tf_dataset.py
import numpy as np
import tensorflow as tf

def load_channel_dataset(data_path, label_path, batch_size=8, shuffle=True, buffer_size=1000):
    # Load .npy files
    data = np.load(data_path)     # Shape: (N, H, W, C)
    labels = np.load(label_path)  # Shape: (N, H, W, C) or (N,) depending on labels

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))

    # Shuffle and batch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset
