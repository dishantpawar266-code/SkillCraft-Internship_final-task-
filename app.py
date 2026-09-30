# ============================================================
# HAND GESTURE RECOGNITION USING CNN
# Dataset: LeapGestRecog
# Kaggle: gti-upm/leapgestrecog
# ============================================================

# ============================================================
# 1. INSTALL LIBRARIES
# ============================================================




# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import kagglehub
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


print("TensorFlow Version:", tf.__version__)


# ============================================================
# 3. DOWNLOAD / IMPORT KAGGLE DATASET
# ============================================================

print("\nDownloading Kaggle dataset...")

path = kagglehub.dataset_download("gti-upm/leapgestrecog")

print("\nDataset downloaded successfully!")
print("Dataset path:")
print(path)


# ============================================================
# 4. FIND THE ACTUAL DATASET DIRECTORY
# ============================================================

# The Kaggle dataset can sometimes contain an extra folder.
# This code automatically searches for the folder containing
# the gesture folders.

print("\nSearching dataset structure...")

gesture_names = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down"
]

dataset_dir = None

for root, dirs, files in os.walk(path):

    found = 0

    for gesture in gesture_names:
        if gesture in dirs:
            found += 1

    if found >= 5:
        dataset_dir = root
        break


if dataset_dir is None:
    print("Could not automatically find gesture folders.")
    print("Searching first-level directories...")

    for item in os.listdir(path):
        full_path = os.path.join(path, item)

        if os.path.isdir(full_path):
            dataset_dir = full_path
            break


print("\nActual dataset directory:")
print(dataset_dir)


# ============================================================
# 5. DISPLAY DATASET CLASSES
# ============================================================

print("\nGesture folders:")

if dataset_dir:

    for item in sorted(os.listdir(dataset_dir)):

        full_path = os.path.join(dataset_dir, item)

        if os.path.isdir(full_path):
            print(" -", item)


# ============================================================
# 6. IMAGE SETTINGS
# ============================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    validation_split=0.20,

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.15,

    shear_range=0.10,

    horizontal_flip=True
)


validation_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    validation_split=0.20
)


# ============================================================
# 8. LOAD TRAINING DATA
# ============================================================

print("\nLoading training images...")

train_data = train_datagen.flow_from_directory(

    dataset_dir,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True,

    seed=SEED
)


# ============================================================
# 9. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation images...")

validation_data = validation_datagen.flow_from_directory(

    dataset_dir,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False,

    seed=SEED
)


# ============================================================
# 10. DISPLAY CLASS NAMES
# ============================================================

class_names = list(train_data.class_indices.keys())

num_classes = len(class_names)

print("\nNumber of classes:", num_classes)

print("\nClass Mapping:")

for class_name, class_number in train_data.class_indices.items():

    print(class_number, "=", class_name)


# ============================================================
# 11. DISPLAY SAMPLE IMAGES
# ============================================================

images, labels = next(train_data)

plt.figure(figsize=(12, 8))

for i in range(12):

    plt.subplot(3, 4, i + 1)

    plt.imshow(images[i])

    label_index = np.argmax(labels[i])

    plt.title(class_names[label_index])

    plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 12. BUILD CNN MODEL
# ============================================================

model = models.Sequential([

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    ),

    layers.MaxPooling2D((2, 2)),


    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),


    # Third convolution block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),


    # Fourth convolution block
    layers.Conv2D(
        256,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),


    # Flatten
    layers.Flatten(),


    # Fully connected layer
    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(0.5),


    # Output layer
    layers.Dense(
        num_classes,
        activation="softmax"
    )

])


# ============================================================
# 13. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 14. COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# ============================================================
# 15. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True
)


checkpoint = ModelCheckpoint(

    "best_hand_gesture_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ============================================================
# 16. TRAIN MODEL
# ============================================================

print("\nStarting model training...\n")

EPOCHS = 15

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        checkpoint
    ]

)


# ============================================================
# 17. EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(validation_data)

print("\nValidation Loss:", loss)

print(
    "Validation Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 18. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 19. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

print("\nGenerating classification report...")

validation_data.reset()

predictions = model.predict(validation_data)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = validation_data.classes

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(12, 10))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=class_names,

    yticklabels=class_names

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(rotation=45)

plt.yticks(rotation=0)

plt.show()


# ============================================================
# 22. SAVE FINAL MODEL
# ============================================================

model.save(
    "hand_gesture_recognition_model.keras"
)

print(
    "\nModel saved as:"
    " hand_gesture_recognition_model.keras"
)


# ============================================================
# 23. SAVE CLASS NAMES
# ============================================================

import json

with open(
    "gesture_classes.json",
    "w"
) as f:

    json.dump(
        class_names,
        f
    )

print(
    "Class names saved as: gesture_classes.json"
)


# ============================================================
# 24. SINGLE IMAGE PREDICTION FUNCTION
# ============================================================

def predict_gesture(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:

        print(
            "ERROR: Image not found."
        )

        return


    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # Resize
    resized = cv2.resize(
        image_rgb,
        IMG_SIZE
    )


    # Normalize
    normalized = resized / 255.0


    # Add batch dimension
    input_image = np.expand_dims(
        normalized,
        axis=0
    )


    # Prediction
    prediction = model.predict(
        input_image,
        verbose=0
    )


    # Get class
    predicted_index = np.argmax(
        prediction[0]
    )


    # Confidence
    confidence = (
        prediction[0][predicted_index] * 100
    )


    # Display
    plt.figure(figsize=(6, 6))

    plt.imshow(image_rgb)

    plt.title(
        f"Gesture: {class_names[predicted_index]}\n"
        f"Confidence: {confidence:.2f}%"
    )

    plt.axis("off")

    plt.show()


    print(
        "Predicted Gesture:",
        class_names[predicted_index]
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )


# ============================================================
# 25. TEST YOUR OWN IMAGE
# ============================================================

# Upload an image in Colab first,
# then replace the filename below.

# Example:
#
# predict_gesture("/content/my_hand.jpg")


# ============================================================
# 26. PROJECT COMPLETE
# ============================================================

print("\n======================================")

print("HAND GESTURE RECOGNITION COMPLETE")

print("======================================")

print(
    "\nModel file:"
    " hand_gesture_recognition_model.keras"
)

print(
    "\nNumber of gesture classes:",
    num_classes
)

print(
    "\nClasses:"
)

for i, name in enumerate(class_names):

    print(
        i,
        "->",
        name
    )

print("\n======================================")