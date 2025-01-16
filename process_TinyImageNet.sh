#!/bin/bash

# Move images from train subdirectories
echo "Processing train directories..."
# cd /data1/anonymous/honeytunnel/data/tiny-imagenet-200/
cd /data1/anonymous/honeytunnel/data/tiny-imagenet-200/
for dr in train/*; do
    echo "Processing $dr..."
    mv "$dr/images"/* "$dr/"
    rmdir "$dr/images"
done

# Move images from val directory based on annotations
cd /data1/anonymous/honeytunnel/data/tiny-imagenet-200/val
echo "Processing val directory..."
while read -r fname label remainder; do
    echo "Moving $fname to val2/$label..."
    mkdir -p "val2/$label"
    mv "images/$fname" "val2/$label/"
done < val_annotations.txt

echo "Process completed."
