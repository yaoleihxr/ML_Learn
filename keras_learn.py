# /usr/bin/python
# -*- encoding:utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# load_Data #
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (batch, height, width, channels)
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
# print(x_train, y_train)
# print(x_train.shape, y_train.shape)
# load_Data #


# model #
model = Sequential()
# layer 2
# 作为模型第一层时，需要提供 input_shape 参数
model.add(Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), input_shape=x_train.shape[1:],
                 data_format='channels_last', padding='same', activation='relu',
                 kernel_initializer='uniform'))
# layer 3
model.add(MaxPooling2D(2,2))
# layer 4
model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), data_format='channels_last',
                 padding='valid', activation='relu', kernel_initializer='uniform'))
# layer 5
model.add(MaxPooling2D(2,2))
# layer 6
model.add(Conv2D(filters=120, kernel_size=(5,5), strides=(1,1), data_format='channels_last',
                 padding='valid', activation='relu', kernel_initializer='uniform'))
model.add(Flatten())
# layer 7
model.add(Dense(84, activation='relu'))
# layer 8
model.add(Dense(10, activation='softmax'))
# print
model.summary()
# model #


# model compile #
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
print('train________________')
model.fit(x_train, y_train, epochs=10, batch_size=128)
print('test_________________')
loss, acc = model.evaluate(x_test, y_test)
print('loss=', loss)
print('accuracy=', acc)
# model compile #


# plot model #
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plot_model(model, to_file='E:/example.png', show_shapes=True)
lena = mpimg.imread('E:/example.png')
print(lena.shape)
plt.imshow(lena)
plt.axis('off')
plt.show()
# plot model #



# save model #
model_path = 'E:/model.h5'
model.save(model_path)

from keras.models import load_model
model = load_model(model_path)
# save model #