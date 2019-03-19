#!/bin/bash

maybe_download () {
  if [ ! -f $1 ]; then
    echo "Downloading $1 ..."
    curl -L "https://static.preferred.ai/vs-cnn/$1" -o $1
  fi
}

# Weights
if [ -d "weights" ]; then
  rm -rf weights
fi
maybe_download "weights.zip"
echo "Extracting weights.zip ..."
unzip -qq weights.zip

echo

# Data
if [ -d "data" ]; then
  rm -rf data
fi
mkdir data

maybe_download "business.zip"
echo "Extracting business.zip ..."
unzip -qq business.zip -d data/

echo

maybe_download "user.zip"
echo "Extracting user.zip ..."
unzip -qq user.zip -d data/

echo

maybe_download ""features.zip""
echo "Extracting features.zip ..."
unzip -qq features.zip -d data/