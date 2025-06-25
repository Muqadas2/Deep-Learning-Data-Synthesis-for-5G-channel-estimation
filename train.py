# train_tf.py
import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

def train(model,model_name, train_dataset, val_dataset=None, epochs=10, lr=3e-4, patience=5, val_freq=1,
             avg_time_per_sample=None, total_params=None):

    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    # Callbacks
    callbacks = []
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True
    )
    callbacks.append(early_stopping_cb)

    log_file = os.path.join(output_dir, f"{model_name}_log.csv")
    csv_logger = tf.keras.callbacks.CSVLogger(log_file)
    callbacks.append(csv_logger)

    print("\nStarting training...\n")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset if val_dataset else None,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print("\nTraining complete.\n")

    model_save_path = os.path.join(output_dir, f"{model_name}_model.keras")
    model.save(model_save_path)
    print(f"Model saved at: {model_save_path}")

    # Save TensorFlow SavedModel directory
    savedmodel_path = os.path.join(output_dir, f"{model_name}_SavedModel")
    model.export(savedmodel_path)

    # Save TFLite model
    tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model saved in Keras format at: {model_save_path}")
    # ---- LOG FILE ----
    log_path = os.path.join(output_dir, "log.txt")
    with open(log_path, "a") as log_file:
        log_file.write(f"\n--- Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_file.write(f"Model: {model_name}\n")
        log_file.write(f"Epochs: {epochs}\n")
        log_file.write(f"Learning Rate: {lr}\n")
        log_file.write(f"Patience: {patience}\n")
        log_file.write(f"Validation Frequency: {val_freq}\n")
        log_file.write(f"Train Loss: {history.history['loss'][-1]:.6f}\n")
        if val_dataset:
            log_file.write(f"Val Loss: {history.history['val_loss'][-1]:.6f}\n")
        log_file.write(f"Early Stopping Triggered: {'Yes' if early_stopping_cb.stopped_epoch > 0 else 'No'}\n")
        log_file.write(f"Total Trainable Parameters: {total_params}\n")
        log_file.write(f"Avg Inference Time per Sample: {avg_time_per_sample:.6f} sec\n")
        log_file.write(f"Model Saved At: {model_save_path}\n")
        log_file.write("-" * 50 + "\n")

    print(f"Log file updated at: {log_path}")

    # ---- SAVE TRAINING PLOT ----
    plot_path = os.path.join(output_dir, f"{model_name}_training_progress.png")
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label="Train Loss")
    if val_dataset:
        plt.plot(history.history['val_loss'], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training & Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Training plot saved as: {plot_path}")

    # ---- SAVE RESULTS TO CSV ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{model_name}_training_results_{timestamp}.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])
        for i in range(len(history.history['loss'])):
            val = history.history['val_loss'][i] if 'val_loss' in history.history else ""
            writer.writerow([i+1, history.history['loss'][i], val])

        writer.writerow([])
        writer.writerow(["Total Trainable Parameters", total_params])
        writer.writerow(["Avg Inference Time per Sample (s)", avg_time_per_sample])
        writer.writerow(["Model Architecture"])
        writer.writerow([model.to_json()])

    print(f"Training results saved to: {csv_path}")

    # ---- APPEND TO GLOBAL SUMMARY FILE ----
    summary_path = os.path.join("results", "summary_all_models.csv")
    summary_exists = os.path.exists(summary_path)

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow([
                "Timestamp", "Model", "Epochs", "LR", "Final Train Loss", "Final Val Loss",
                "Early Stopped", "Total Params", "Avg Inference Time (s)", "Model Path"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            epochs,
            lr,
            round(history.history['loss'][-1], 6),
            round(history.history['val_loss'][-1], 6) if 'val_loss' in history.history else "N/A",
            "Yes" if early_stopping_cb.stopped_epoch > 0 else "No",
            total_params,
            round(avg_time_per_sample, 6),
            model_save_path
        ])
        print(f"Global summary updated at: {summary_path}")
