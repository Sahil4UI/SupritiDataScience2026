import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageOps

# model load kro
model = tf.keras.models.load_model("handwriting_model.keras")

# dataset=labels
LABELS = list("0123456789")+list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")+list("abcdefghijklmnopqrstuvwxyz")


# image ko model ke input format me lao
def preprocess_image(image_path):
    # color BGR -> Black n white
    image = Image.open(image_path).convert("L")

    # image ke pixels unkoarray me convert kro
    pixels = np.array(image)

    # white bg hai to invert kro
    if np.mean(pixels)>127:
        image = ImageOps.invert(image)

    pixels = np.array(image)

    #background noise/blurr ko km krna h
    pixels = np.where(pixels > 50, 255,0).astype(np.uint8)
    image = Image.fromarray(pixels)

    # character ke aspas ke extra bg ko crop kro
    bbox = image.getbbox()
    if bbox is None:
        raise ValueError("Image is Blank")
    image = image.crop(bbox)

    # character uske aspas square
    width,height = image.size
    side = max(width,height) +8 

    canvas = Image.new("L",(side,side),color=0)
    canvas.paste(
        image,
        ((side-width)//2,(side-height)//2)
    )

    # 20X20 me resize krke 28X28 pr rakhna h
    canvas = canvas.resize((20,20),Image.Resampling.LANCZOS)

    final_image = Image.new("L",(28,28),color=0)
    final_image.paste(canvas,(4,4))

    array = np.array(final_image,dtype=np.float32) / 255.0

    # model ko 4D input chaiye
    # batch 1
    array = array.reshape(1,28,28,1)

    return array , final_image


image_path = "test_images/image.png"

try:
    processed_image,display_image = preprocess_image(image_path)
    # plt.imshow(processed_image,cmap="gray")
    # plt.title("Image Given to Model")
    # plt.axis("off")
    # plt.show()
    predictions = model.predict(processed_image,verbose=0)[0]

    # sabse high score wali class choose kro
    index = int(np.argmax(predictions))
    character = LABELS[index]
    score = float(predictions[index])*100

    print("\n==========RESULT===========")
    print("Predicted Character : ",character)
    print("Predicted Score : ",round(score,2),"%")

    # top 3 predictions
    top_indices = np.argsort(predictions)[-3:][::-1]

    print("\nTOP 3 Predictions : ")
    for i in top_indices:
        print(f"{LABELS[int(i)]} : {predictions[i]*100:.2f}%")

except FileNotFoundError:
    print("FIle Not Found")

except ValueError as error:
    print("Image Error")


    