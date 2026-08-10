import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

print(f"TensorFlow: {tf.__version__}")
print(f"GPUs encontradas: {gpus}")

if not gpus:
    raise RuntimeError("O TensorFlow não encontrou nenhuma GPU.")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)