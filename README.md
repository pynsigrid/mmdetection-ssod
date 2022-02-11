# mmdetection-ssod
## basic train
1. create config file
```
cd mmdetection/work_dirs/riceblast
python create_config.py
```
2. train
```
python tools/train.py work_dirs/riceblast/faster_rcnn_r50_caffe_fpn_1x_coco_job1/faster_rcnn_r50_caffe_fpn_1x_coco_job1.py --gpu-ids 0
```
3. 