#!/bin/bash

# Create a temporary filelist.txt
filelist="filelist.txt"
rm -f $filelist

# Loop through all mp4 files in the current directory and add them to filelist.txt
for f in *.mp4; do
  echo "file '$f'" >> $filelist
done

# Combine the files using ffmpeg
ffmpeg -f concat -safe 0 -i $filelist -c copy output.mp4

# Clean up
rm -f $filelist

echo "All videos have been combined into output.mp4"
