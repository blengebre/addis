import tensorflow as tf

model = tf.keras.models.load_model(
    "amh_omo_transformer.keras",
    compile=False
)

model.summary()

print(model.inputs)