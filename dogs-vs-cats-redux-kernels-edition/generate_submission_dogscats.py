from __future__ import unicode_literals

import re
import csv
import os

from vgg16 import Vgg16
from utils import save_array, load_array, get_data, get_batches, onehot
import numpy as np
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop

MODEL_PATH = 'models/'
BATCHES_FILENAME = MODEL_PATH + 'train_data.bc'
VAL_BATCHES_FILENAME = MODEL_PATH + 'val_data.bc'

CLASSES_BATCHES_FILENAME = MODEL_PATH + 'classes_train_data.bc'
CLASSES_VAL_BATCHES_FILENAME = MODEL_PATH + 'classes_val_data.bc'

TRAIN_LASTLAYER_FEATURES = MODEL_PATH + 'train_lastlayer_features.bc'
VAL_LASTLAYER_FEATURES = MODEL_PATH + 'valid_lastlayer_features.bc'


def write_csv(data):
    with open('submission.csv', 'wb') as f:
        w = csv.writer(f)
        w.writerow(['id', 'label'])
        keys = sorted(data.keys())
        for key in keys:
            w.writerow([key, "%.5f" % data[key]])


def main():
    path = 'sample/' # In order to use the big dataset set it to empty string.
    batch_size = 1 # If you are using a GPU, please change it to 64
    # We are going to use the 1000 categories from the IMAGENET model,
    # so we need to re-use an instance of it.
    vgg = Vgg16()
    # First, we check if we have saved arrays with the data and classes.
    if (os.path.exists(BATCHES_FILENAME) and
            os.path.exists(VAL_BATCHES_FILENAME) and
            os.path.exists(CLASSES_BATCHES_FILENAME) and
            os.path.exists(CLASSES_VAL_BATCHES_FILENAME)):
        print 'Loading arrays...'
        train_data = load_array(BATCHES_FILENAME)
        val_data = load_array(VAL_BATCHES_FILENAME)
        batches_classes = load_array(CLASSES_BATCHES_FILENAME)
        val_classes = load_array(CLASSES_VAL_BATCHES_FILENAME)
    else:
        # We get the images from the different files and resize them so
        # VGG16 can process them (224, 224). Also we need the classes.

        # As first step we need to get the batches with category, so we can
        # infer the classes later on.
        print 'Calculating arrays...'
        batches = get_batches(path+'train', batch_size=batch_size)
        val_batches = get_batches(path+'valid', batch_size=batch_size)
        # After that, we get the classes.
        batches_classes = batches.classes
        val_classes = val_batches.classes
        # Finally, once we have the classes get need to fetch the images again
        # so the classes param is NONE and in only ONE batch.
        train_data = get_data(path+'train')
        val_data = get_data(path+'valid')
        save_array(BATCHES_FILENAME, train_data)
        save_array(VAL_BATCHES_FILENAME, val_data)
        save_array(CLASSES_BATCHES_FILENAME, batches_classes)
        save_array(CLASSES_VAL_BATCHES_FILENAME, val_classes)

    # We need the labels in "hot encoding form". e.g [1, 0] or [0, 1]
    train_labels = onehot(batches_classes)
    val_labels = onehot(val_classes)

    vgg.model.summary()
    # Now we need to create the model.
    print "Let's remove the last layer as it's already DENSE..."
    vgg.model.pop()
    # Here we say that we only want to train the last layer
    for layer in vgg.model.layers:
        layer.trainable = False
    print 'And add a new layer that outputs only the two classes (dogs or cats)'
    vgg.model.add(Dense(2, activation='softmax'))
    vgg.model.compile(optimizer=RMSprop(lr=0.1),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    vgg.model.summary()

    vgg.model.fit(train_data, train_labels, nb_epoch=1,
                  validation_data=(val_data, val_labels), batch_size=batch_size)
    print 'Saving weights'
    vgg.model.save_weights(MODEL_PATH + 'weights.bc')
    vgg.model.load_weights(MODEL_PATH + 'weights.bc')

    test_batches, predictions = vgg.test(path+'test', batch_size=batch_size)

    d = {}
    for idx, filename in enumerate(test_batches.filenames):
        result = int(re.search('%s(.*)%s' % ('\/', '\.'), filename).group(1))
        # We use a trick to never show 0 or 1, but 0.05 and 0.95.
        # This is required becase log loss penalizes predictions that are confident and wrong.
        d[result] = predictions[idx][1].clip(min=0.05, max=0.95)
    write_csv(d)

if __name__ == '__main__':
    main()
