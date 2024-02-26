torchrun --nproc_per_node=2 train.py \
--data VOC.yaml --epochs 100 --weights yolov5s.pt --cfg yolov5s.yaml  --batch-size 192 --device 1,2