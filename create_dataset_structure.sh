#!/bin/bash

# Please place this script in the new competition's folder and only after
# the dataset have been downloaded and uncompressed.

# Create symbolic links
ln -s ../vgg16.py .
ln -s ../vgg16bn.py .
ln -s ../utils.py .
ln -s ../toolbox.py .

# Create esential folders (we assume that "train" and "test" are already there)
mkdir test/unknown
mkdir sample
mkdir sample/train
mkdir sample/test
mkdir sample/test/unknown
mkdir sample/valid
mkdir sample/models
mkdir sample/files
mkdir valid
mkdir models
mkdir files

ROOT_DIR=$(pwd)
cd ${ROOT_DIR}

DIRS=`ls -l train | egrep '^d' | awk '{print $9}'`

# Move all the test data inside "test/unknown"
mv test/* test/unknown

# Populate "valid" folder
for d in $DIRS ; do
    mkdir valid/${d}
    cd train/${d}
    shuf -zn1000 -e *.jpg | xargs -0  mv -vt ../../valid/${d}
    cd ${ROOT_DIR}
done

# Populate "sample" folder
for d in $DIRS ; do
    mkdir sample/train/${d}
    mkdir sample/valid/${d}
    cd train/${d}
    shuf -zn100 -e *.jpg | xargs -0  cp -vt ../../sample/train/${d}
    cd ../../sample/train/${d}
    shuf -zn50 -e *.jpg | xargs -0  mv -vt ../../valid/${d}
    cd ${ROOT_DIR}
done
cd test/unknown
shuf -zn200 -e *.jpg | xargs -0  cp -vt ../../sample/test/unknown
cd ${ROOT_DIR}
