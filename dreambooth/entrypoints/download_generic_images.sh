#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <subdirectory of repo> <path to output directory>"
    exit 1
fi

# Download pre-computed images of the generic from subdirectory of the github repo
DIR=$1
OUT_DIR=$2
URL="https://github.com/jbpacker/dreambooth_class_images/trunk/$DIR"

mkdir -p $OUT_DIR
cd $OUT_DIR
svn checkout $URL .
