# 3DMPB-dataset

Dataset proposed in "Pose2UV: Single-shot Multi-person Mesh Recovery with Deep UV Prior"  
[[ProjectPage]](https://www.yangangwang.com/papers/HBZ-Pose2UV-2022-06.html) [[paper]](https://www.yangangwang.com/papers/HBZ-pose2uv-2022-06.pdf) @ TIP2022. 

<div align="center">
<img src="docs/image.png" width="800px"/>
<p> Samples of 3DMPB Dataset</p>
</div>

**3DMPB** is a multi-person dataset in the outdoor sport field with human interaction occlusion and image truncation. This dataset provides comprehensive annotations including bounding-box, human 2D pose, SMPL model annotations, instance mask and camera parameters.


## Statistics
**Note:** **Not** all the instances in this dataset are annotated in consideration of some inaccurate annotations or wrong relative occlusion. 

|          | bbox    | keypoint   | mask  | SMPL |
| -------- | :-----: | :--------: | :----: | :-----:|
| Images   | 13,665  | 13,665     | 13,665 | 13,665 |
| Persons  | 25,122  | 25,122     | 25,122 | 25,122 |

## Download Links

- [Images & Masks & Annotations]()

## Installation
### Requirements 
* python3
* numpy
* pickle
* trimesh
* pyrender
### Prepare dataset 
Please download the dataset above and extract under `./3DMPB`. Due to the licenses, please download SMPL model file [here](https://smpl.is.tue.mpg.de/). And make them look like this:
```
|-- 3DMPB
`-- |-- images
    |   |-- 000000.jpg
    |   |-- 000001.jpg
    |   |-- 000002.jpg
    |   |-- ...
    |-- masks
    |   |-- 000000_00.jpg
    |   |-- 000000_01.jpg
    |   |-- 000000_02.jpg
    |   |-- 000001_00.jpg
    |   |-- 000002_00.jpg
    |   |-- ...
    |-- annot.json
|-- data
`-- |-- SMPL_NEUTRAL.pkl    
```
### Code installation
#### visualize 2D keypoints and boundingbox
```
git clone https://github.com/boycehbz/3DMPB-dataset.git
cd/d 3DMPB-dataset
python vis_dataset.py --dataset_dir 3DMPB  --output_dir output 
```
#### visualize 3D meshes
```
git clone https://github.com/boycehbz/3DMPB-dataset.git
cd/d 3DMPB-dataset
python vis_dataset.py --dataset_dir 3DMPB  --output_dir output --vis_smpl True
```

