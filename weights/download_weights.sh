#!/bin/bash
# Download weights for vanilla YOLOv3
wget -c https://pjreddie.com/media/files/yolov3.weights --no-check-certificate
# # Download weights for tiny YOLOv3
wget -c https://pjreddie.com/media/files/yolov3-tiny.weights --no-check-certificate
# Download weights for backbone network
wget -c https://pjreddie.com/media/files/darknet53.conv.74 --no-check-certificate
