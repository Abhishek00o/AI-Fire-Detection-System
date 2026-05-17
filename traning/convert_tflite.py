import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("../model/fire_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# Save TFLite model
with open("../model/fire_model.tflite", "wb") as f:
    f.write(tflite_model)

print("🔥 TFLite model created successfully!")