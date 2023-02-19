#!/bin/bash

# Prompt the user to enter options for script1.py
echo "Enter video path: "
read options1
python frame_extract.py --video-path $options1 &

# Prompt the user to enter options for script2.py
echo "Enter images path: "
read options2
python face_align.py --images-path $options2 &

# Prompt the user to enter options for script3.py
echo "Enter images, landmarks and boxes paths: "
read options3, options4, options5
python eye_mouth_crop.py --images-path $options3 --landmarks-path $options4 --boxes-path $options5&
