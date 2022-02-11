import os
import random
import numpy as np
import torch
from mmcv import Config
from mmdet.apis import set_random_seed

seed = 7777

"""Sets the random seeds."""
set_random_seed(seed, deterministic=False)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
# cwd_path = '/home/pai/workspace/mmdetection'

job_num = '1'  # for version control
model_name = f'faster_rcnn_r50_caffe_fpn_1x_coco_job{job_num}'  
work_dir = os.path.join(os.getcwd(), model_name)  
baseline_cfg_path = "configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco_job.py"  
# print('******', baseline_cfg_path)

cfg_path = os.path.join(work_dir, model_name + '.py')  

train_data_images = os.getcwd() + '/../../data/RiceBlast.v10/train2017'  
val_data_images = os.getcwd() + '/../../data/RiceBlast.v10/val2017'  
test_data_images = os.getcwd() + '/../../data/RiceBlast.v10/test2017'  

# File config
num_classes = 3  
classes = ("RiceBlast-MR","RiceBlast-MS","RiceBlast-R")  
# model zoo: https://github.com/open-mmlab/mmdetection/blob/master/README_zh-CN.md
load_from = 'open-mmlab://detectron2/resnet50_caffe'

train_ann_file = os.getcwd() + '/../../data/RiceBlast.v10/instances_train2017.json'  
val_ann_file = os.getcwd() + '/../../data/RiceBlast.v10/instances_val2017.json'  

# Train config              
gpu_ids = [1]  
total_epochs = 30  
batch_size = 2 ** 2  # 建议是2的倍数。
num_worker = 2  # 比batch_size小就行
log_interval = 100  
checkpoint_interval = 8  
evaluation_interval = 1  
lr = 0.01 / 2  

# TODO: add extra configs manual

def create_mm_config():
    cfg = Config.fromfile(baseline_cfg_path)

    cfg.work_dir = work_dir

    # Set seed thus the results are more reproducible
    cfg.seed = seed

    # You should change this if you use different model
    cfg.load_from = load_from

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    print("| work dir:", work_dir)

    # Set the number of classes
    # for head in cfg.model.roi_head.bbox_head:
    #     head.num_classes = num_classes
    cfg.model.roi_head.bbox_head.num_classes = num_classes  # TODO-0211: create user config 

    cfg.gpu_ids = gpu_ids

    cfg.runner.max_epochs = total_epochs  # Epochs for the runner that runs the workflow
    cfg.total_epochs = total_epochs

    # Learning rate of optimizers. The LR is divided by 8 since the config file is originally for 8 GPUs
    cfg.optimizer.lr = lr

    ## Learning rate scheduler config used to register LrUpdater hook
    cfg.lr_config = dict(
        policy='CosineAnnealing',
        # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
        by_epoch=False,
        warmup='linear',  # The warmup policy, also support `exp` and `constant`.
        warmup_iters=500,  # The number of iterations for warmup
        warmup_ratio=0.001,  # The ratio of the starting learning rate used for warmup
        min_lr=1e-07)

    # config to register logger hook
    cfg.log_config.interval = log_interval  # Interval to print the log

    # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    cfg.checkpoint_config.interval = checkpoint_interval  # The save interval is 1

    cfg.dataset_type = 'CocoDataset'  # Dataset type, this will be used to define the dataset
    cfg.classes = classes

    cfg.data.train.img_prefix = train_data_images
    cfg.data.train.classes = cfg.classes
    cfg.data.train.ann_file = train_ann_file
    cfg.data.train.type = 'CocoDataset'

    cfg.data.val.img_prefix = val_data_images
    cfg.data.val.classes = cfg.classes
    cfg.data.val.ann_file = val_ann_file
    cfg.data.val.type = 'CocoDataset'

    cfg.data.test.img_prefix = val_data_images
    cfg.data.test.classes = cfg.classes
    cfg.data.test.ann_file = val_ann_file
    cfg.data.test.type = 'CocoDataset'

    cfg.data.samples_per_gpu = batch_size  # Batch size of a single GPU used in testing
    cfg.data.workers_per_gpu = num_worker  # Worker to pre-fetch data for each single GPU

    # The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
    cfg.evaluation.metric = 'bbox'  # Metrics used during evaluation

    # Set the epoch intervel to perform evaluation
    cfg.evaluation.interval = evaluation_interval

    cfg.evaluation.save_best = 'bbox_mAP'

    cfg.log_config.hooks = [dict(type='TextLoggerHook')]

    print("| config path:", cfg_path)
    # Save config file for inference later
    cfg.dump(cfg_path)
    # print(f'CONFIG:\n{cfg.pretty_text}')


if __name__ == '__main__':
    print("—" * 50)
    # print('******', 'base pwd: ', os.getcwd())

    create_mm_config()
    print("—" * 50)
