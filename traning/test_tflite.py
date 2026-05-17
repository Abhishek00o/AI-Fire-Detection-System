import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="../model/fire_model.tflite"
)

interpreter.allocate_tensors()

print("✅ TFLite model working successfully!")