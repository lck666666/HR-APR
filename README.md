# HR-APR: APR-agnostic Framework with Uncertainty Estimation and Hierarchical Refinement for Camera Relocalisation
This is the official repo of paper "HR-APR: APR-agnostic Framework with Uncertainty Estimation and Hierarchical Refinement for Camera Relocalisation"

### [Homepage](https://lck666666.github.io/research/HR-APR/index.html) | [Paper](https://arxiv.org/pdf/2402.14371.pdf)

**HR-APR: APR-agnostic Framework with Uncertainty Estimation and Hierarchical Refinement for Camera Relocalisation** <br>
[Changkun Liu](https://lck666666.github.io)<sup>1</sup>, [Shuai Chen](https://chenusc11.github.io/)<sup>3</sup>, [Yukun Zhao](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=NcLael4AAAAJ)<sup>2</sup>,
[Huajian Huang](https://huajianup.github.io/)<sup>1</sup>, [Victor Prisacariu](https://www.robots.ox.ac.uk/~victor/)<sup>3</sup> and [Tristan Braud](https://braudt.people.ust.hk/index.html)<sup>1,2</sup> <br>
HKUST CSE<sup>1</sup>, HKUST ISD<sup>2</sup>, Active Vision Lab, University of Oxford<sup>3</sup> <br>
**International Conference on Robotics and Automation (ICRA) 2024**<br>
![framework](images/framework.png)
![uncertainty](images/uncertainty_estimation_module.png)
## Usage
```
git clone https://github.com/lck666666/HR-APR.git
cd HR-APR
pip install json numpy matplotlib
```

## Show HR-APR results in the paper
We already prepared the `txt` results files for three APRs:`DFNet, MS-Transformer, PoseNet` in `APR/7Scenes` and `APR/Cambridge`. We provide `scene_test_gt.txt, scene_train_gt.txt, scene_predict.txt, scene_Nefes_N.txt`, where `N=10,30,50` presents running the refinement process for `N` iterations. We provide the similarity score of each images in `APR/simi_ranking`. 
```
cd uncertainty_module
python hr_apr_7s.py --apr DFNet --scene fire --gamma 0.95
python hr_apr_cam.py --apr DFNet --scene KingsCollege --gamma 0.96
```
For all pose files, the format is 
```image_name x y z qw qx qy qz```.
```x, y, z``` are camera to world coordinates.
## Visualization
You can generate the Fig.4 in the main paper by executing
```
cd visualization
python errorChange7s.py --apr DFNet --gamma 0.95
python errorChangeCam.py --apr DFNet --gamma 0.95
```
You can generate the Fig.5 in the main paper by executing
```
cd visualization
python plotSimiErrorRatio.py --apr DFNet
python plotSimiErrorRot --apr DFNet
python plotSimiErrorTrans --apr DFNet
```

## Try the whole pipeline from scratch
We release the uncertainty module and visualization code in this repo. For feature extractor depicted in the paper, you can check the readme of [PoseNet-Pytorch](https://github.com/youngguncho/PoseNet-Pytorch) and use the code in our ```./feature_extractor```, to do the feature extraction for PoseNet model, download the datasets [Cambridge Landmarks](https://paperswithcode.com/dataset/cambridge-landmarks) and [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/), put datasets in ```./feature_extractor/posenet```, then train a posenet model (also a feature extractor), see line 245 in ```./feature_extractor/model.py```
```
cd ./feature_extractor
python train.py --image_path ./posenet/KingsCollege --metadata_path ./posenet/KingsCollege/dataset_train.txt --model Efficientnet
```
then, the feature descriptors are stored in ```.npy``` format as shown in the line 318 of ```./feature_extractor/solver.py``` by running ```test.py```. 
```
python test.py --image_path ./posenet/KingsCollege --metadata_path ./posenet/KingsCollege/dataset_test.txt --model Efficientnet
```

To generate the same files in `APR/simi_ranking`, you can execute 
```
python cal_7s_simi.py --apr DFNet --scene fire
python cal_cam_simi.py --apr DFNet --scene KingsCollege
```
To get Nefes refinement predictions of each APR, you can check the CVPR2024 [paper](https://github.com/ActiveVisionLab/NeFeS).


## Acknowledgement
Our feature extractor implementation is referenced and modified from the reproduced PoseNet code [here](https://github.com/youngguncho/PoseNet-Pytorch?tab=readme-ov-file). Thanks [@youngguncho](https://github.com/youngguncho) for the excellent work!

## Citation
Please cite our paper and star this repo if you find our work helpful. Thanks!
```
@inproceedings{liu2024hrapr,
title = {HR-APR: APR-agnostic Framework with Uncertainty Estimation and Hierarchical Refinement for Camera Relocalisation},
author={Changkun Liu and Shuai Chen and Yukun Zhao and Huajian Huang and Victor Prisacariu and Tristan Braud},
booktitle = {International Conference on Robotics and Automation (ICRA)},
year = {2024},
organization={IEEE}
}
```
If you are also interested in the pose refinement module, please cite
```
@inproceedings{chen2024nefes,
  author    = {Chen, Shuai and Bhalgat, Yash and Li, Xinghui and Bian, Jia-Wang and Li, Kejie and Wang, Zirui and Prisacariu, Victor Adrian},
  title     = {Neural Refinement for Absolute Pose Regression with Feature Synthesis},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {20987-20996}
}
```
