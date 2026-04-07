import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
DATASET_DIR = "dataset"

train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255,
    validation_split   = 0.2,
    horizontal_flip    = True,
    zoom_range         = 0.2,
    rotation_range     = 15,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
)

val_datagen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    validation_split = 0.2,
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = "binary",
    subset      = "training",
    seed        = 42,
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = "binary",
    subset      = "validation",
    seed        = 42,
)

print(f" Training:   {train_generator.samples} images")
print(f" Validation: {val_generator.samples} images")
print(f" Classes:    {train_generator.class_indices}")

# ── STEP 2: LOAD MobileNetV2 BASE ─────────────────────────
print("\n Loading MobileNetV2 base model...")

base_model = MobileNetV2(
    weights      = "imagenet",   # pre-trained on 1.2M images
    include_top  = False,        # remove original classifier
    input_shape  = (IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model — we don't retrain MobileNetV2's layers
# We only train our custom layers on top
base_model.trainable = False

print(f" MobileNetV2 loaded — {len(base_model.layers)} layers frozen")


x = base_model.output
x = GlobalAveragePooling2D()(x)     # compress features
x = Dense(128, activation="relu")(x) # learn mask patterns
x = Dropout(0.5)(x)                  # prevent overfitting
x = Dense(64, activation="relu")(x)  # deeper understanding
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)  # final: mask or no mask

model = Model(inputs=base_model.input, outputs=output)

print(f" Total layers     : {len(model.layers)}")
print(f" Trainable layers : {len([l for l in model.layers if l.trainable])}")

print("\n  Compiling model...")

model.compile(
    optimizer = Adam(learning_rate=0.0001),
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"]
)

print(" Model compiled!")

# ── STEP 5: CALLBACKS (Smart Training Features) ───────────
print("\n  Setting up smart training callbacks...")

# 1. Save the best model automatically
checkpoint = ModelCheckpoint(
    filepath          = "models/best_model.keras",
    monitor           = "val_accuracy",
    save_best_only    = True,
    verbose           = 1,
)

# 2. Stop training early if no improvement (saves time)
early_stop = EarlyStopping(
    monitor   = "val_accuracy",
    patience  = 5,           # stop if no improvement for 5 epochs
    verbose   = 1,
    restore_best_weights = True,
)

# 3. Reduce learning rate when stuck
reduce_lr = ReduceLROnPlateau(
    monitor  = "val_loss",
    factor   = 0.5,          # cut learning rate in half
    patience = 3,
    verbose  = 1,
    min_lr   = 1e-7,
)


print("\n Starting training...")
print("=" * 50)

history = model.fit(
    train_generator,
    epochs            = EPOCHS,
    validation_data   = val_generator,
    callbacks         = [checkpoint, early_stop, reduce_lr],
)

print("=" * 50)
print(" Training complete!")

print("\n Plotting training history...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history["accuracy"],     label="Train Accuracy",      color="blue")
ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange")
ax1.set_title("Model Accuracy Over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history["loss"],     label="Train Loss",      color="blue")
ax2.plot(history.history["val_loss"], label="Validation Loss", color="orange")
ax2.set_title("Model Loss Over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()
print(" Training graph saved as training_history.png")

# ── STEP 8: FINAL EVALUATION ──────────────────────────────
print("\n Evaluating on validation set...")
loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f"\n Final Validation Accuracy : {accuracy * 100:.2f}%")
print(f" Final Validation Loss     : {loss:.4f}")

print("\n Step 4 Complete! Model saved to models/best_model.keras")