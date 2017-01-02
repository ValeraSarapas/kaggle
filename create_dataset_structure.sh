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
mkdir valid
mkdir models

ROOT_DIR=$(pwd)
cd ${ROOT_DIR}

DIRS=`ls -l train | egrep '^d' | awk '{print $9}'`

# Move all the test data inside "test/unknown"
mv test/* test/unknown

# Populate "valid" folder
for d in $DIRS ; do
    mkdir valid/${d}
    cd train/${d}
    shuf -zn500 -e *.jpg | xargs -0  mv -vt ../../valid/${d}
    cd ${ROOT_DIR} 
done

# Populate "sample" folder
for d in $DIRS ; do
    mkdir sample/train/${d}
    mkdir sample/valid/${d}
    cd train/${d}
    shuf -zn40 -e *.jpg | xargs -0  cp -vt ../../sample/train/${d}
    cd ../../sample/train/${d}
    shuf -zn10 -e *.jpg | xargs -0  mv -vt ../../sample/valid/${d}
    cd ${ROOT_DIR} 
done
cd test/unknown
shuf -zn100 -e *.jpg | xargs -0  mv -vt ../../sample/test/unknown
cd ${ROOT_DIR} 