
<div align="center">
<h2>SDGOCC: Semantic and Depth-Guided Bird's-Eye View Transformation for 3D Multimodal Occupancy Prediction (CVPR 2025)</h2>
</div>



## 🚀 Model Zoo  
[Model Weights](https://pan.baidu.com/s/1OSoMYKUfrGTYrP2Ufm-q-Q?pwd=uky8) 


## 🙏 Acknowledgement

This repository is fork from  [SDGOCC: Semantic and Depth-Guided Bird's-Eye View Transformation for 3D Multimodal Occupancy Prediction (CVPR 2025)
](https://github.com/DzpLab/SDGOCC)


## 📂 Known Issues

1. `.npy` file process is not released by the author. I must conduct the process, which may cause errors. If you have idea about how to process the `.npy` file, please contact me or submit a pr. In this step, the error demonstrate like `FileNotFoundError: [Errno 2] No such file or directory: 'data/nuscenes/point_label/LIDAR_TOP/n015-2018-07-11-11-54-16+0800__LIDAR_TOP__1531281439800013.npy'`

2. My env is different from FlashOcc and the author's. I use CUDA 11.7 and Ubuntu22.04.

3. I clone FlashOcc conda env to SDGOCC env, so I can't provide exact env config step.

## 🛠️ Result

mIoU is obviously lower than the original paper, which may be caused by the above issues. The result is as follows:
```
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 5.9 task/s, elapsed: 1016s, ETA:     0s
Starting Evaluation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6019/6019 [00:21<00:00, 283.13it/s]
===> per class IoU of 6019 samples:
===> others - IoU = 0.0
===> barrier - IoU = 0.0
===> bicycle - IoU = 0.03
===> bus - IoU = 0.0
===> car - IoU = 0.0
===> construction_vehicle - IoU = 0.02
===> motorcycle - IoU = 0.0
===> pedestrian - IoU = 0.09
===> traffic_cone - IoU = 0.01
===> trailer - IoU = 0.0
===> truck - IoU = 0.08
===> driveable_surface - IoU = 3.59
===> other_flat - IoU = 0.0
===> sidewalk - IoU = 0.0
===> terrain - IoU = 0.0
===> manmade - IoU = 1.95
===> vegetation - IoU = 0.0
===> mIoU of 6019 samples: 0.34
===>voxel_occ - IoU = 12.26
{'mIoU': 0.34}
```

In next stage, I will discover how to generate correct `.npy` file and try to reproduce the result in the paper.

If you have and idea about the usage of the `.npy` file, please contact me or submit a pr. Thanks a lot!

## 📃 Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@inproceedings{duan2025sdgocc,
  title={SDGOCC: Semantic and Depth-Guided Bird's-Eye View Transformation for 3D Multimodal Occupancy Prediction},
  author={Duan, ZaiPeng and Dang, ChenXu and Hu, Xuzhong and An, Pei and Ding, Junfeng and Zhan, Jie and Xu, YunBiao and Ma, Jie},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={6751--6760},
  year={2025}
}
```
