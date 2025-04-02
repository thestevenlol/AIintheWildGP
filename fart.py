import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# --- Configuration ---
CSV_DATA_FILE = "gesture_landmark_data.csv"
NUM_LANDMARKS = 21
NUM_COORDS = 3 # x, y, z
INPUT_FEATURES = NUM_LANDMARKS * NUM_COORDS # Should be 63

# Training parameters
TEST_SPLIT_RATIO = 0.2 # Use 20% of data for testing/validation
RANDOM_SEED = 6648949 
EPOCHS = 1 # Landmarks often train faster, adjust as needed
BATCH_SIZE = 128

# Expected class names (ensure these match what's in your CSV 'label' column)
# This order will determine the final output mapping
EXPECTED_CLASS_NAMES = ['up', 'down', 'forward', 'back', 'stop', 'land']
NUM_CLASSES = len(EXPECTED_CLASS_NAMES)

MODEL_SAVE_PATH = 'hand_gesture_model_landmarks.keras'
# --- End Configuration ---

# 1. Load Data
try:
    df = pd.read_csv(CSV_DATA_FILE)
    print(f"Loaded data: {df.shape[0]} samples")
    print("Data sample:\n", df.head())
    print("\nGesture Counts:\n", df['label'].value_counts())
except FileNotFoundError:
    print(f"Error: Data file not found at {CSV_DATA_FILE}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Verify number of features
expected_num_columns = 1 + INPUT_FEATURES # Label + features
if df.shape[1] != expected_num_columns:
    print(f"Error: Expected {expected_num_columns} columns, but found {df.shape[1]} in CSV.")
    print("Check if data collection saved correctly (label + 63 landmark coords).")
    exit()


# 2. Prepare Data
# Separate features (landmarks) and labels
X = df.iloc[:, 1:].values # Features (all columns except the first 'label' column)
y_labels = df.iloc[:, 0].values # Labels (the first 'label' column)

# Check unique labels detected vs expected
detected_labels = sorted(list(df['label'].unique()))
print("\nDetected Labels:", detected_labels)
if set(detected_labels) != set(EXPECTED_CLASS_NAMES):
    print("*** WARNING ***: Detected labels don't match EXPECTED_CLASS_NAMES.")
    print("Expected:", sorted(EXPECTED_CLASS_NAMES))
    print("Ensure your collected data covers all expected gestures and uses the correct names.")
    # Update EXPECTED_CLASS_NAMES and NUM_CLASSES based on detection if proceeding
    # EXPECTED_CLASS_NAMES = detected_labels
    # NUM_CLASSES = len(EXPECTED_CLASS_NAMES)
    # print(f"Adjusted to use detected labels. NUM_CLASSES = {NUM_CLASSES}")


# Encode labels (string names to integer indices 0, 1, 2...)
# Use LabelEncoder and fit it ONLY on the expected names to ensure consistent mapping
label_encoder = LabelEncoder()
label_encoder.fit(EXPECTED_CLASS_NAMES) # Fit on the predefined list
y = label_encoder.transform(y_labels) # Transform the actual labels

print("\nLabels encoded:")
for i, name in enumerate(label_encoder.classes_):
     print(f"{name} -> {i}")

# Save the label encoder classes for use during prediction
np.save('landmark_label_classes.npy', label_encoder.classes_)
print("\nSaved label mapping to landmark_label_classes.npy")

# Split data into Training and Validation/Test sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED, stratify=y # Stratify ensures proportional representation
)

print(f"\nTraining data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")

# 3. Build the MLP Model
model = keras.Sequential([
    layers.Input(shape=(INPUT_FEATURES,)),
    layers.Dense(128, activation='relu'), # WAS 128
    layers.Dropout(0.2), # Maybe increase dropout if simplifying
    layers.Dense(64, activation='relu'), # WAS 64
    # layers.Dropout(0.4), # Maybe only one dropout layer is needed
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# 4. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Use sparse CE for integer labels
              metrics=['accuracy'])

# Optional: Add Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 5. Train the Model
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping], # Add early stopping
    verbose=2 # Show one line per epoch
)
print("Training finished.")

# 6. Evaluate the Model
print("\nEvaluating model on validation data...")
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

# Detailed classification report
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report (Validation Set):\n")
# Use target_names from the fitted LabelEncoder for report clarity
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

# 7. Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 8. Save the Model
print(f"\nSaving trained model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved.")