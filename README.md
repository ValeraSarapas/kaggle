# Kaggle Competitions

This repository contains the solutions I created during the course [fast.ai](http://course.fast.ai)
about deep learning. Same of the generic files have been provided by the instructors: e.g `utils.py`, `vgg16.py`...


## Set up Environment

* 1. Install Anaconda from (here)[https://www.continuum.io/downloads]
* 2. Update it: `conda update conda && conda update --all`
* 3. Install aux libs: `conda install mingw libpython bcolz`
* 4. Install Theano: `pip install git+git://github.com/Theano/Theano.git`
* 5. Install Keras: `pip install git+git://github.com/fchollet/keras.git`

## Download Datasets

If order to download the datasets from kaggle, please install the `kaggle-cli` and follow the
instructions. The `generate_submission.py` files assume that you have the correct folders structure per each competition directory.

## Automatic set up of competition folders structure

To setup the correct folders structure just copy the `create_dataset_structure.py` into a folder with the `train` and `test`
directories containing the images and execute it.

## Run & access Jupyter notebooks

* 1. Run `jupyter notebook`
* 2. Go to http://localhost:8000
