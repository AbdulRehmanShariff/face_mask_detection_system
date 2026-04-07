import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE   = 224
BATCH_SIZE = 32
DATASET_DIR = "dataset"




train_datagen = ImageDataGenerator(
    rescale           = 1.0 / 255,       # normalize
    validation_split  = 0.2,             # 20% for validation
    horizontal_flip   = True,            # randomly flip images
    zoom_range        = 0.2,             # randomly zoom
    rotation_range    = 15,              # randomly rotate ±15°
    width_shift_range = 0.1,             # slight horizontal shift
    height_shift_range= 0.1,             # slight vertical shift
)

val_datagen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    validation_split = 0.2,
)

print(" Loading Training Data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = BATCH_SIZE,
    class_mode   = "binary",             
    subset       = "training",           
    seed         = 42,
)

print(" Loading Validation Data...")
val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = BATCH_SIZE,
    class_mode   = "binary",
    subset       = "validation",       
    seed         = 42,
)

print(f"\n  Class Mapping : {train_generator.class_indices}")
print(f" Training images  : {train_generator.samples}")
print(f" Validation images: {val_generator.samples}")
print(f" Batch size       : {BATCH_SIZE}")
print(f" Train batches    : {len(train_generator)}")
print(f" Val batches      : {len(val_generator)}")

print("\n Generating sample image grid...")

images, labels = next(train_generator)
class_names = {v: k for k, v in train_generator.class_indices.items()}

plt.figure(figsize=(15, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i])
    title = "With Mask " if labels[i] == 0 else "No Mask ❌"
    plt.title(title, fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png")
plt.show()
print(" sample_images.png saved!")

