"""
TensorFlow / Keras CNN Pipeline for Intel Image Classification
Architecture: Custom CNN with BatchNorm, Dropout, and Global Average Pooling
Saved to: your_firstname_model.keras
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    CSVLogger
)

# ─────────────────────────────────────────────────────────────
# 1.  CONSTANTS
# ─────────────────────────────────────────────────────────────

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet mean/std (same as PyTorch pipeline for consistency)
_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


# ─────────────────────────────────────────────────────────────
# 2.  DATA PIPELINE
# ─────────────────────────────────────────────────────────────

def _find_folder(root, candidates):
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        f"Could not find any of {candidates} inside '{root}'. "
        "Please check --data_dir."
    )


def _normalize(image, label):
    """Scale [0,255]→[0,1] then apply ImageNet normalisation."""
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - _MEAN) / _STD
    return image, label


def _augment(image, label):
    """Random augmentations applied only during training."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    # Random rotation via tf.keras preprocessing layer equivalent
    image = _random_rotate(image, max_angle_deg=15)
    return image, label


def _random_rotate(image, max_angle_deg=15):
    """Rotate image by a random angle in [-max_angle_deg, +max_angle_deg]."""
    import math
    max_angle_rad = max_angle_deg * math.pi / 180.0
    angle = tf.random.uniform([], -max_angle_rad, max_angle_rad)
    image = tf.keras.ops.image.affine_transform(
        tf.expand_dims(image, 0),
        transform=_get_rotation_matrix(angle),
        interpolation="bilinear",
        fill_mode="reflect",
    )[0] if hasattr(tf.keras.ops, 'image') else _rotate_fallback(image, angle)
    return image


def _rotate_fallback(image, angle):
    """Fallback rotation using tfa or skipping if unavailable."""
    try:
        import tensorflow_addons as tfa
        return tfa.image.rotate(image, angle, interpolation="BILINEAR")
    except ImportError:
        # Skip rotation if tfa is not available
        return image


def _get_rotation_matrix(angle):
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    return tf.stack([cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0], axis=0)


def get_datasets(data_dir: str, img_size: int, batch_size: int):
    """
    Build train / val / test tf.data.Dataset pipelines.
    """
    train_dir = _find_folder(data_dir, ["seg_train/seg_train", "seg_train", "train"])
    test_dir  = _find_folder(data_dir, ["seg_test/seg_test",  "seg_test",  "test"])

    # ── Load from directory ────────────────────────────────────
    raw_train = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=None,          # infer from subdirectory names
        image_size=(img_size, img_size),
        batch_size=None,           # batch manually after augmentation
        shuffle=True,
        seed=42,
        validation_split=0.1,
        subset="training",
    )

    raw_val = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=None,
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=False,
        seed=42,
        validation_split=0.1,
        subset="validation",
    )

    raw_test = keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=None,
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=False,
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (raw_train
                .map(_augment,   num_parallel_calls=AUTOTUNE)
                .map(_normalize, num_parallel_calls=AUTOTUNE)
                .batch(batch_size)
                .prefetch(AUTOTUNE))

    val_ds = (raw_val
              .map(_normalize, num_parallel_calls=AUTOTUNE)
              .batch(batch_size)
              .prefetch(AUTOTUNE))

    test_ds = (raw_test
               .map(_normalize, num_parallel_calls=AUTOTUNE)
               .batch(batch_size)
               .prefetch(AUTOTUNE))

    # Count samples
    n_train = sum(1 for _ in raw_train)
    n_val   = sum(1 for _ in raw_val)
    n_test  = sum(1 for _ in raw_test)

    print(f"\n[Data] Train samples : {n_train}")
    print(f"[Data] Val   samples : {n_val}")
    print(f"[Data] Test  samples : {n_test}")

    return train_ds, val_ds, test_ds, n_train


# ─────────────────────────────────────────────────────────────
# 3.  MODEL DEFINITION
# ─────────────────────────────────────────────────────────────

def conv_block(x, filters, pool=False, l2=1e-4):
    """Conv2D → BN → ReLU (→ optional MaxPool2D)."""
    x = layers.Conv2D(
        filters, (3, 3), padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2),
        kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if pool:
        x = layers.MaxPooling2D((2, 2))(x)
    return x


def build_intel_cnn(img_size: int = 150, num_classes: int = 6) -> keras.Model:
    """
    Mirror architecture of the PyTorch IntelCNN:
      Stage 1 :  3 → 32 → 64   (img_size → img_size/2)
      Stage 2 : 64 → 128 → 128 (       → /4)
      Stage 3 :128 → 256 → 256 (       → /8)
      Stage 4 :256 → 512 → 512 (       → /16)
      Head    : GAP → Dense(256, relu) → Dropout(0.4) → Dense(num_classes)
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")

    # Stage 1
    x = conv_block(inputs, 32)
    x = conv_block(x, 64, pool=True)

    # Stage 2
    x = conv_block(x, 128)
    x = conv_block(x, 128, pool=True)

    # Stage 3
    x = conv_block(x, 256)
    x = conv_block(x, 256, pool=True)

    # Stage 4
    x = conv_block(x, 512)
    x = conv_block(x, 512, pool=True)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = keras.Model(inputs, outputs, name="IntelCNN")
    return model


# ─────────────────────────────────────────────────────────────
# 4.  TRAINING HELPERS
# ─────────────────────────────────────────────────────────────

def plot_history(history: keras.callbacks.History, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    acc_key = "accuracy" if "accuracy" in history.history else "acc"
    axes[1].plot(history.history[acc_key],         label="Train Acc")
    axes[1].plot(history.history[f"val_{acc_key}"], label="Val Acc")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[TensorFlow] Training plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 5.  MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_tensorflow_pipeline(args):
    # ── GPU memory growth ──────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device_name = "GPU" if gpus else "CPU"
    print(f"\n[TensorFlow] Using device: {device_name}")
    print(f"[TensorFlow] TF version  : {tf.__version__}")

    # ── Data ──────────────────────────────────────────────────
    train_ds, val_ds, test_ds, n_train = get_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    # ── Model ─────────────────────────────────────────────────
    model = build_intel_cnn(img_size=args.img_size, num_classes=NUM_CLASSES)
    model.summary()

    # ── Compile ───────────────────────────────────────────────
    # Label smoothing is applied via the loss function (mirrors PyTorch setup)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
    )

    # Cosine-decay learning rate schedule
    steps_per_epoch = max(1, n_train // args.batch_size)
    total_steps     = steps_per_epoch * args.epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=1e-6,
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # ── Callbacks ─────────────────────────────────────────────
    best_ckpt_path = os.path.join(args.output_dir, "best_tf_checkpoint.keras")
    callbacks = [
        ModelCheckpoint(
            filepath=best_ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(os.path.join(args.output_dir, "tf_training_log.csv")),
    ]

    # ── Training ──────────────────────────────────────────────
    print(f"\n[TensorFlow] Starting training for {args.epochs} epochs …\n")
    t0 = time.time()

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - t0
    print(f"\n[TensorFlow] Training finished in {elapsed/60:.1f} min")

    # ── Test evaluation ───────────────────────────────────────
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n[TensorFlow] Test Accuracy : {test_acc*100:.2f}%")
    print(f"[TensorFlow] Test Loss     : {test_loss:.4f}")

    # ── Save final model ──────────────────────────────────────
    save_path = os.path.join(args.output_dir, "your_firstname_model.keras")
    model.save(save_path)
    print(f"[TensorFlow] Model saved → {save_path}")

    # ── Plot ──────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, "tensorflow_training_history.png")
    plot_history(history, plot_path)

    best_val_acc = max(history.history.get("val_accuracy",
                                           history.history.get("val_acc", [0])))
    print(f"\n[TensorFlow] Best Val Acc : {best_val_acc*100:.2f}%")
    print(f"[TensorFlow] Test Acc     : {test_acc*100:.2f}%")
    return model
