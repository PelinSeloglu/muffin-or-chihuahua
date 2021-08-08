import numpy as np
import os
import cv2
import random
import pickle

DATAPATH = "muffin or chihuahua"

CATEGORIES = ["chihuahua", "muffin"]

#fotoğrafların boyutu
IMGS = 200

#dosyada bulunan fotoğrafların kontrolü
for category in CATEGORIES:
    path = os.path.join(DATAPATH, category)

    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATAPATH, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMGS, IMGS))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

X = []  # features
y = []  # labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMGS, IMGS, 1)
y = np.array(y)

# veri seti ile ilgili bilgileri barındıran dosyaların oluşturulması
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)