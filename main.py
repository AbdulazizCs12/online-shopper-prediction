import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


data = pd.read_csv("online_shoppers_intention.csv")

print("Dataset Shape:", data.shape)
print("Missing Values:", data.isnull().sum().sum())

y = data["Revenue"].astype(int)
X = data.drop("Revenue", axis=1)

categorical_columns = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeatures after encoding:", X.shape[1])
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    0: class_weights_array[0],
    1: class_weights_array[1]
}

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            threshold=0.0,
            name="accuracy"
        )
    ]
)

print("\nNeural Network Architecture:")
model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

logits = model.predict(X_test)

probabilities = tf.nn.sigmoid(logits).numpy()

y_pred = (probabilities >= 0.5).astype(int).flatten()

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Purchase", "Purchase"]
))

plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

print("\nTraining plots saved successfully.")