import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

model = tf.keras.models.load_model("CNN.model")

DATAPATH = "muffin or chihuahua"
IMGS = 200
category = "test"

path = os.path.join(DATAPATH, category)

for img in os.listdir(path):

    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    img_array2 = np.asarray(Image.open(os.path.join(path, img)))
    new_array = cv2.resize(img_array, (IMGS, IMGS))
    new_array = np.array(new_array).reshape(-1, IMGS, IMGS, 1)
    # img_path_name = path + "/" + img
    prediction = model.predict(new_array)

    if prediction[0][0] == 1:

        plt.xlabel('Tahminlenen --> Bir Chihuahuadır.', fontsize=16)
        plt.imshow(img_array2)
        plt.show()

    elif prediction[0][1] == 1:

        plt.xlabel('Tahminlenen --> Bir Muffindir.', fontsize=16)
        plt.imshow(img_array2)
        plt.show()