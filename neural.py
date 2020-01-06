from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np


def train_and_check_model():
    train_dir = 'train'
    val_dir = 'val'

    img_width, img_height = 224, 224
    epochs = 2
    batch_size = 5
    nb_train_samples = 460
    nb_validation_samples = 80

    # vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #
    # vgg19.trainable = False
    #
    # model = Sequential()
    # model.add(vgg19)
    # model.add(Flatten())
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4))
    # model.add(Activation('sigmoid'))

    model = load_model('sky_neural.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=10)


    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)


    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save('sky_neural.h5')

    plt.plot(history.history['accuracy'],
             label='Доля верных ответов на обучающем наборе')
    plt.plot(history.history['val_accuracy'],
             label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()


def predicting_model(name_img):
    model = load_model('sky_neural.h5')

    img = image.load_img(name_img,
                         target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    result = model.predict(x)

    return result