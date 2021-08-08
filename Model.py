from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import matplotlib.pyplot as plt

# veriler ile ilgili dosyaların açılması
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# pixellerin normalleştirilmesi( 0’dan 255’ e)
X = X/255.0

# modelin oluşturulması
model = Sequential()
# 3 convolutional katmanı oluşturulması
model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2 hidden katmanı oluşturulması
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

# 2 sınıf için 2 nöronlu output katmanı oluşturma
model.add(Dense(2))
model.add(Activation("softmax"))

# modelin derlenmesi
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# modelin 32 yineleme ile eğitilmesi
history = model.fit(X, y, batch_size=32, epochs=12, validation_split=0.12)

# modelin kayıt edilmesi
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

model.save('CNN.model')

# modelin eğitimi sırasında doğruluk oranında ki değişimi gösteren grafiğin bastırılması
print(history.history.keys())
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Doğruluğu')
plt.ylabel('Doğruluk')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()