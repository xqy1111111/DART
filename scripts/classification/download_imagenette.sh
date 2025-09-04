#!/usr/bin/env bash
set -euo pipefail

# Download ImageNette (160px) and unpack into data/imagenette2-160

DATA_DIR=${1:-data}
SIZE=${2:-160}

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ "$SIZE" = "160" ]; then
  URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
elif [ "$SIZE" = "320" ]; then
  URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
else
  URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
fi

FNAME=$(basename "$URL")

if [ ! -f "$FNAME" ]; then
  echo "Downloading $URL ..."
  wget -q "$URL"
else
  echo "File $FNAME already exists, skipping download"
fi

echo "Extracting $FNAME ..."
tar xf "$FNAME"

echo "Done. Dataset is under: $(pwd)/${FNAME%.tgz}"

