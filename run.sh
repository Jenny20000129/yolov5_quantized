# python train.py --data coco_local.yaml --epochs 300 --weights yolov5s.pt --cfg yolov5s.yaml  --batch-size 64 --device 2 --factorq 2.5
#python train.py --data 54.yaml --epochs 10 --weights yolov5s.pt --cfg yolov5s.yaml  --batch-size 32 --device 2 --factorq 1
#python train.py --data VOC.yaml --epochs 300 --weights '' --cfg yolov5s.yaml  --batch-size 64 --device 2 --factorq 1
python train.py --data pinpu.yaml --epochs 300 --weights runs/train/success24_pinpu_withoutQ/weights/last.pt --cfg yolov5s.yaml  --batch-size 16 --device 1 --factorq 2.5
python train.py --data pinpu_new.yaml --epochs 300 --weights yolov5s.pt --cfg yolov5s.yaml  --batch-size 16 --device 1 --factorq 0
python train.py --data pinpu_new.yaml --epochs 300 --weights /home/huyaqi/yolov5/runs/train/success38_pinpuNew_withoutQ/weights/last.pt --cfg yolov5s.yaml  --batch-size 16 --device 1 --factorq 2.5
python train.py --data pinpu_new.yaml --epochs 1000 --weights runs/train/success24_pinpu_withoutQ/weights/last.pt --cfg yolov5s.yaml  --batch-size 16 --device 1 --factorq 2.5