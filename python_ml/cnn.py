import tensorflow as tf

from tensorflow.keras import datasets, layers, models, regularizers, constraints
import matplotlib.pyplot as plt

import numpy as np
import io
import os
import struct

tf.enable_eager_execution()

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

plt.figure(figsize=(10, 10))

if not os.path.exists('images.dat'):
    with open('images.dat', mode='wb') as f:
        for i in range(test_images.shape[0]):
            f.write(struct.pack('d', test_labels[i][0]))  # put all labels into file consecutively
        for i in range(test_images.shape[0]):
            for x in test_images[i].ravel():  # get values
                f.write(struct.pack('d', (x + 1.) / 2.))  # encode values, shifted to convert from bipolar to unipolar
        f.close()

dropout = 0.1  # training dropout ratio
factors = [4., 15., 20., 25., 35.]  # Btanh scaling factors for each layer
r_values = [6, 4, 8, 8, 4]  # Btanh r values for each layer - TODO possible improvement: use Btanh with given r as training activation function, instead of true tanh

model = models.Sequential()

if os.path.exists('model'):
    model = models.load_model('model')
else:
    model.add(layers.Conv2D(20, (3, 3), padding='same', strides=1, kernel_initializer='he_uniform',
                            kernel_constraint=constraints.max_norm(1. / factors[0], []),
                            bias_constraint=constraints.max_norm(1. / factors[0], []),
                            activation='tanh', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout))
    model.add(layers.Conv2D(40, (3, 3), padding='same', strides=1, kernel_initializer='he_uniform',
                            kernel_constraint=constraints.max_norm(1. / factors[1], []),
                            bias_constraint=constraints.max_norm(1. / factors[1], []),
                            #kernel_regularizer=regularizers.l2(regu_param_c), bias_regularizer=regularizers.l2(regu_param_c),
                            activation='tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout))
    model.add(layers.Conv2D(60, (3, 3), padding='same', strides=1, kernel_initializer='he_uniform',
                            kernel_constraint=constraints.max_norm(1. / factors[2], []),
                            bias_constraint=constraints.max_norm(1. / factors[2], []),
                            activation='tanh'))
    model.add(layers.Conv2D(60, (3, 3), padding='same', strides=1, kernel_initializer='he_uniform',
                            kernel_constraint=constraints.max_norm(1. / factors[3], []),
                            bias_constraint=constraints.max_norm(1. / factors[3], []),
                            activation='tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, kernel_initializer='he_uniform',
                           kernel_constraint=constraints.max_norm(1. / factors[4], []),
                           bias_constraint=constraints.max_norm(1. / factors[4], []),
                           activation='tanh'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(train_images, train_labels, epochs=50, batch_size=64,
                        validation_data=(test_images, test_labels))

    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    model.save('model')

model.summary()
model.evaluate(test_images, test_labels, verbose=2)
model.evaluate(test_images[0:100], test_labels[0:100], verbose=2)

layers = [model.layers[0], model.layers[3], model.layers[6], model.layers[7], model.layers[11], model.layers[13]]  # layers with weights

out_strings = []

for i in range(len(layers) - 1):
    layer = layers[i]
    factor = factors[i]
    r_val = r_values[i]
    kern = layer.kernel.numpy()
    out_strings.append(','.join([str(factor)] + [str(r_val)] + [str((factor * x + 1.) / 2.) for x in kern.ravel()]))  # shift from bipolar to unipolar values
    out_strings.append(','.join([str(factor)] + [str(r_val)] + [str((factor * x + 1.) / 2.) for x in layer.bias.numpy()]))

# handle last layer (software) separately, no shift, no scale factor
layer = layers[-1]
kern = layer.kernel.numpy()
out_strings.append(','.join([str(0. + x) for x in kern.ravel()]))
out_strings.append(','.join([str(0. + x) for x in layer.bias.numpy()]))

with open('weights.csv', mode='w') as f:
    f.write('\n'.join(out_strings))
    f.close()
