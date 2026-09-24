import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder

TRAIN_PATH = "PlantVillage/Corn (Maize)/Train"
VAL_PATH = "PlantVillage/Corn (Maize)/Val"
TEST_PATH = "PlantVillage/Corn (Maize)/Test"

IMAGE_SIZE = (128, 128)

def load_dataset(folder):
    features = []
    labels = []

    classes = sorted(os.listdir(folder))

    print(f"\nLoading images from: {folder}")

    for cls in classes:

        class_folder = os.path.join(folder, cls)

        if not os.path.isdir(class_folder):
            continue

        print(f"Class : {cls}")

        files = os.listdir(class_folder)

        print(f"Number of images: {len(files)}")

        for filename in files:

            image_path = os.path.join(class_folder, filename)

            # Read image
            image = cv2.imread(image_path)

            # Skip unreadable images
            if image is None:
                print(f"Warning: Unable to read image {image_path}")
                continue

            # Resize image
            image = cv2.resize(
                image,
                IMAGE_SIZE
            )

            # Convert image to grayscale
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # Extract HOG features
            hog_features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

            # Store features and label
            features.append(hog_features)
            labels.append(cls)

    print(f"Successfully loaded: {len(features)} images")

    return np.array(features), np.array(labels)

print("\nLoading Training Dataset...")
X_train, y_train = load_dataset(TRAIN_PATH)
print("\nLoading Validation Dataset...")
X_val, y_val = load_dataset(VAL_PATH)

X_train = np.concatenate((X_train, X_val))
y_train = np.concatenate((y_train, y_val))
print("\nTraining Images: ", len(X_train))

print("\nLoading Testing Dataset...")
X_test, y_test = load_dataset(TEST_PATH)
print("\nTesting Images: ", len(X_test))

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
print("\nClasses Found")
for i, label in enumerate(encoder.classes_):
    print(i, "->", label)
print("\nFeature Vector Size :", X_train.shape[1])
print("\nDataset Loaded Successfully!")

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("\nTraining SVM model...")

svm_model = SVC(kernel='rbf', C=10, probability = True, random_state = 42)
svm_model.fit(X_train, y_train)

print("SVM training completed!")

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")

print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

display.plot(
    cmap="Blues",
    xticks_rotation=45
)

plt.title("Maize Disease Classification - Confusion Matrix")

plt.tight_layout()

plt.show()

joblib.dump(
    svm_model,
    "maize_svm_model.pkl"
)

joblib.dump(
    encoder,
    "maize_label_encoder.pkl"
)

# ============================================================
# VERIFY SAVED MODEL
# ============================================================

print("\n==============================")
print("VERIFYING SAVED MODEL")
print("==============================")

# Reload model and encoder
loaded_svm = joblib.load("maize_svm_model.pkl")
loaded_encoder = joblib.load("maize_label_encoder.pkl")

# Predict entire test dataset again
saved_predictions = loaded_svm.predict(X_test)

saved_accuracy = accuracy_score(
    y_test,
    saved_predictions
)

print(
    f"\nSaved Model Test Accuracy: "
    f"{saved_accuracy * 100:.2f}%"
)

# Check whether both models give identical predictions
same_predictions = np.array_equal(
    y_pred,
    saved_predictions
)

print(
    "Original and saved predictions identical:",
    same_predictions
)

# ============================================================
# TEST ONE EXACT IMAGE
# ============================================================

test_folder = "PlantVillage/Corn (Maize)/Test/Common Rust"

for filename in os.listdir(test_folder):

    image_path = os.path.join(
        test_folder,
        filename
    )

    image = cv2.imread(image_path)

    if image is not None:
        break

print("\n==============================")
print("INDIVIDUAL IMAGE TEST")
print("==============================")

print("Actual Class: Common Rust")
print("Image:", filename)

# Resize
image = cv2.resize(
    image,
    IMAGE_SIZE
)

# Grayscale
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# HOG
features = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)

features = features.reshape(1, -1)

print("HOG Feature Size:", features.shape)

# Prediction
prediction = loaded_svm.predict(
    features
)[0]

predicted_class = loaded_encoder.inverse_transform(
    [prediction]
)[0]

print("Predicted Class:", predicted_class)

# Probabilities
probabilities = loaded_svm.predict_proba(
    features
)[0]

print("\nClass Probabilities:")

for class_name, probability in zip(
    loaded_encoder.classes_,
    probabilities
):
    print(
        f"{class_name}: "
        f"{probability * 100:.2f}%"
    )
    
print("\nModel saved as: maize_svm_model.pkl")

print("Label encoder saved as: maize_label_encoder.pkl")