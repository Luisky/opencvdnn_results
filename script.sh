#!/bin/bash

sync
echo 3 > /proc/sys/vm/drop_caches

mkdir -p /sys/fs/cgroup/memory/OPENCVDNN
echo $1 >> /sys/fs/cgroup/memory/OPENCVDNN/memory.limit_in_bytes
cat /sys/fs/cgroup/memory/OPENCVDNN/memory.limit_in_bytes

blktrace/blktrace -d /dev/vdb1 -o dnn &

cd /home/vm/opencvdnn_results/build
./Detection --config=../yolo_files/yolov3.cfg --model=../yolo_files/yolov3.weights --classes=../yolo_files/coco.names --width=416 --height=416 --scale=0.00392 --input=../yolo_files/fall.mp4 --rgb
