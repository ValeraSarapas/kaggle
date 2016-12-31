from __future__ import unicode_literals, division, print_function

import numpy as np
import re
import os

from vgg16 import Vgg16
from utils import save_array, load_array, get_data, get_batches, onehot
from toolbox import write_submission_csv, extract_images_and_classes, MODELS_PATH
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K


def main():
    path = 'sample/' # In order to use the big dataset set it to empty string.
    batch_size = 1 # If you are using a GPU, please change it to 64
    # We are going to use the 1000 categories from the IMAGENET model,
    # so we need to re-use an instance of it.
    vgg = Vgg16()
    # First, we check if we have saved arrays with the data and classes.
    train_data, val_data, batches_classes, val_classes = extract_images_and_classes(
        path, batch_size=batch_size
    )

    # We need the labels in "hot encoding form". e.g [1, 0] or [0, 1]
    train_labels = onehot(batches_classes)
    val_labels = onehot(val_classes)

    # It's always good to see an overview of the whole model.
    vgg.model.summary()
    # Now we need to create the model.
    # --------------------------------------------------------
    # 0) Replace and train the last Dense layer as it doesn't really
    # fit our purposes.
    print('Let\'s remove the last layer as it\'s already DENSE...')
    vgg.model.pop()
    # Here we say that we only want to train the last layer
    for layer in vgg.model.layers: layer.trainable = False
    print('And train a new layer that outputs only the two classes (dogs or cats)')
    vgg.model.add(Dense(2, activation='softmax'))
    opt = RMSprop(lr=0.1)
    vgg.model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    #vgg.model.fit(train_data, train_labels, nb_epoch=1,
    #              validation_data=(val_data, val_labels), batch_size=batch_size)
    # --------------------------------------------------------
    # 1) Backpropagate and train the previous layers.
    print('Now let\'s train the previous layers using backpropagation')
    layers = vgg.model.layers
    # Get the index of the first dense layer...
    first_dense_idx = [index for index, layer in enumerate(layers) if type(layer) is Dense][0]
    # ...and set this and all subsequent layers to trainable
    for layer in layers[first_dense_idx:]: layer.trainable = True
    K.set_value(opt.lr, 0.01)
    #vgg.model.fit(train_data, train_labels, nb_epoch=1,
    #              validation_data=(val_data, val_labels), batch_size=batch_size)

    print('Saving weights')
    vgg.model.save_weights(MODELS_PATH + 'weights.bc')
    #vgg.model.load_weights(MODELS_PATH + 'weights.bc')

    test_batches, predictions = vgg.test(path+'test', batch_size=batch_size)

    d = {}
    for idx, filename in enumerate(test_batches.filenames):
        result = int(re.search('%s(.*)%s' % ('\/', '\.'), filename).group(1))
        # We use a trick to never show 0 or 1, but 0.05 and 0.95.
        # This is required becase log loss penalizes predictions that are confident and wrong.
        d[result] = predictions[idx][1].clip(min=0.05, max=0.95)
    write_submission_csv(d, ['id', 'label'])

if __name__ == '__main__':
    main()
