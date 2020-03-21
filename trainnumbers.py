import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import cv2

#tf version: 1.8.0
#keras version: 2.1.5

K.set_image_dim_ordering('th') #sets depth, input_depth, rows, columns for the convolutional neural network

raw_data = pd.read_csv("emnist-mnist-train.csv")

all_X = raw_data.values[:,1:]
all_y = raw_data.values[:,0]

all_X = np.array(all_X)
all_y = np.array(all_y)

X = all_X.reshape(all_X.shape[0], 1, 28, 28).astype('float32') #3d array
y = all_y

#displays random letters
# for i in range(15):
#     index = random.randint(0, len(all_X) - 1)
#     random_letter = all_X[index].reshape(28, 28, 1).astype('float32') #3d array
#
#     random_letter = cv2.rotate(random_letter, cv2.ROTATE_90_CLOCKWISE)
#     random_letter = cv2.flip(random_letter, 1)
#
#     random_letter = cv2.resize(random_letter, (400, 400))
#     cv2.imshow(chr(64 + all_y[index]), random_letter)
#     cv2.waitKey(0)

# normalize inputs from 0-255 to 0-1
X = X / 255

y = np_utils.to_categorical(y)
num_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


def basic_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(1, 28, 28), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    #regular neural network part
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile models
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# #train

model = basic_model()

# # Fit the models
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=500, verbose=1)
# Final evaluation of the models
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


# #save
#
# serialize models to YAML
model_yaml = model.to_yaml()
with open("models/number.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("models/number.h5")

print("Saved models to disk")
