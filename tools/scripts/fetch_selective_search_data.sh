#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../data" && pwd )"
cd $DIR

FILE=selective_search_data.tgz
URL=https://www.dropbox.com/s/g1z7iolrtxdo56c/selective_search_data.tgz?dl=0
CHECKSUM=7cc85568609e1ac645f102c37eb376c3

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

echo "Downloading precomputed selective search boxes (1.8G)..."

wget $URL -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
