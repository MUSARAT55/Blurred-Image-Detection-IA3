
import os, tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE=(128,128); BATCH=32
train=tf.keras.utils.image_dataset_from_directory(
 "dataset/train", image_size=IMG_SIZE, batch_size=BATCH,
 label_mode="binary", class_names=["clear","blurred"])
val=tf.keras.utils.image_dataset_from_directory(
 "dataset/validation", image_size=IMG_SIZE, batch_size=BATCH,
 label_mode="binary", class_names=["clear","blurred"])

model=models.Sequential([
 layers.Rescaling(1./255,input_shape=(128,128,3)),
 layers.Conv2D(32,3,activation="relu"),layers.MaxPooling2D(),
 layers.Conv2D(64,3,activation="relu"),layers.MaxPooling2D(),
 layers.Conv2D(128,3,activation="relu"),layers.MaxPooling2D(),
 layers.Flatten(),layers.Dense(128,activation="relu"),
 layers.Dropout(.3),layers.Dense(1,activation="sigmoid")
])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(train,validation_data=val,epochs=10)
os.makedirs("model",exist_ok=True)
model.save("model/blur_detector.h5")
print("Saved model/blur_detector.h5")
