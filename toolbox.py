from __future__ import unicode_literals, division, print_function

import csv
import os
import numpy as np
from utils import save_array, load_array, get_data, get_batches, onehot


MODELS_PATH = 'models/'
BATCHES_FILENAME = MODELS_PATH + 'train_data.bc'
VAL_BATCHES_FILENAME = MODELS_PATH + 'val_data.bc'

CLASSES_BATCHES_FILENAME = MODELS_PATH + 'classes_train_data.bc'
CLASSES_VAL_BATCHES_FILENAME = MODELS_PATH + 'classes_val_data.bc'


def extract_images_and_classes(path, batch_size=1):
    """
    Utility function specialized in fetching images and
    classes from disk. It also caches them after calculation.
    """
    if (os.path.exists(BATCHES_FILENAME) and
            os.path.exists(VAL_BATCHES_FILENAME) and
            os.path.exists(CLASSES_BATCHES_FILENAME) and
            os.path.exists(CLASSES_VAL_BATCHES_FILENAME)):
        print('Loading arrays from disk...')
        train_data = load_array(BATCHES_FILENAME)
        val_data = load_array(VAL_BATCHES_FILENAME)
        batches_classes = load_array(CLASSES_BATCHES_FILENAME)
        val_classes = load_array(CLASSES_VAL_BATCHES_FILENAME)
    else:
        # We get the images from the different files and resize them so
        # VGG16 can process them (224, 224). Also we need the classes.

        # As first step we need to get the batches with category, so we can
        # infer the classes later on.
        print('Calculating arrays as they could not be found in disk ...')
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

    return train_data, val_data, batches_classes, val_classes

