import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import keras

# random seed se results ko reproductible bana skte hai
np.random.seed(42)
tf.random.set_seed(42)

# model me total 62 character classes hai
NUM_CLASSES = 62

# trained model MODEL PATH
MODEL_PATH = "handwriting_model.keras"

# step2 : handwritting dataset loading
print("Loading handwritting dataset...")

(train_ds,test_ds) , info = tfds.load(
    "emnist/byclass",
    split = ["train","test"],
    as_supervised=True,
    with_info=True
)

print("Dataset Loaded!")
print("Original Training images : ",info.splits["train"].num_examples)
print("Original Testing images : ",info.splits["test"].num_examples)

# Images ko prepare karna hai
def preprocess_image(image,label):
    # Pixel 0-255 -> 0to1
    image = tf.cast(image, tf.float32) / 255.0

    # EMNIST images -> orientation 
    image = tf.transpose(image, perm = [1,0,2]) 

    return image,tf.cast(label, tf.int32)

# demo keliye dataset ke subset lene hai
# data shuffle - alag alag classes k sample
TRAIN_LIMIT = 100_000
TEST_LIMIT = 20_000
BATCH_SIZE = 128

train_ds = (
    # autotune - har image label pairs uspe preprocess function chalega-
    # parallell processing - AUTOTUNE
    train_ds.shuffle(100_000,seed=42).take(TRAIN_LIMIT).map(preprocess_image,num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    test_ds.shuffle(20_000,seed=42).take(TEST_LIMIT).map(preprocess_image,num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)

# CNN model building
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28,28,1)),
        # 28 height, 28 width, 1 channel - > grayscale
        tf.keras.layers.Conv2D(
            32,kernel_size=3,activation="relu"
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        # features - maps ke 2X2 area me max value uthayega
        tf.keras.layers.Conv2D(
             64,kernel_size=(3,3),activation="relu"
        ),
         tf.keras.layers.MaxPool2D(pool_size=(2,2)),

         tf.keras.layers.Flatten(),
        # 128 neurons unke learned features h unko combine krna h
        tf.keras.layers.Dense(128,activation="relu"),

        tf.keras.layers.Dense(NUM_CLASSES,activation="softmax")
        # har possible character keliye output probability dega
    ]
)


model.compile(
    optimizer="adam",
    # adam optimizer - training k time model ke weights update krta h
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

model.summary()

print("Training Started...") 

history = model.fit(
    train_ds,
    epochs = 3,
    validation_data = test_ds
)

print("Training Completed...")

# testing data uspe model ko check krna h
test_loss,test_accuracy = model.evaluate(test_ds,verbose=1)
# verbose- progress
print("Test Loss : ",round(test_loss),4)
print("Test Accuracy : ",round(float(test_accuracy)*100,2),"%")


model.save(MODEL_PATH)

print("Model saved : ", MODEL_PATH)

# step 10 visualization : 
plt.figure(figsize=(8,4))

plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Handwritting Recognition Training")
plt.legend()
plt.show()

