#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../data" && pwd )"
cd $DIR

FILE=fast_rcnn_models.tgz
# TODO
URL=
# TODO
CHECKSUM=

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

# TODO
echo "Downloading precomputed selective search boxes (1.8G)..."

wget $URL -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
