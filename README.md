# Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching

This contains the codes for cross-view geo-localization method described in: Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching, CVPR2020. 
![alt text](./Framework.png)

# Abstract
Cross-view geo-localization is the problem of estimating the position and orientation (latitude, longitude and azimuth angle) of a camera at ground level given a large-scale database of geo-tagged aerial (\eg, satellite) images.
Existing approaches treat the task as a pure location estimation problem by learning discriminative feature descriptors, but neglect orientation alignment. 
It is well-recognized that knowing the orientation between ground and aerial images can significantly reduce matching ambiguity between these two views, especially when the ground-level images have a limited Field of View (FoV) instead of a full field-of-view panorama.
Therefore, we design a Dynamic Similarity Matching network to estimate cross-view orientation alignment during localization. 
In particular, we address the cross-view domain gap by applying a polar transform to the aerial images to approximately align the images up to an unknown azimuth angle. Then, a two-stream convolutional network is used to learn deep features from the ground and polar-transformed aerial images.
Finally, we obtain the orientation by computing the correlation between cross-view features, which also provides a more accurate measure of feature similarity, improving location recall.
Experiments on standard datasets demonstrate that our method significantly improves state-of-the-art performance.
Remarkably, we improve the top-1 location recall rate on the CVUSA dataset by a factor of  $1.5\times$ for panoramas with known orientation, 
by a factor of  $3.3\times$ for panoramas with unknown orientation, and by a factor of  $6\times$ for $180^{\circ}$-FoV images with unknown orientation.
### Experiment Dataset
We use two existing dataset to do the experiments

- CVUSA dataset: a dataset in America, with pairs of ground-level images and satellite images. All ground-level images are panoramic images.  
	The dataset can be accessed from https://github.com/viibridges/crossnet

- CVACT dataset: a dataset in Australia, with pairs of ground-level images and satellite images. All ground-level images are panoramic images.  
	The dataset can be accessed from https://github.com/Liumouliu/OriCNN


### Dataset Preparation: Polar transform
1. Please Download the two datasets from above links, and then put them under the director "Data/". The structure of the director "Data/" should be:
"Data/CVUSA/
 Data/ANU_data_small/"
2. Please run "data_preparation.py" to get polar transformed aerial images of the two datasets and pre-crop-and-resize the street-view images in CVACT dataset to accelerate the training speed.


### Codes
Codes for training and testing on unknown orientation (train_grd_noise=360) and different FoV.

1. Training:
	CVUSA: python train_cvusa_fov.py --polar 1 --train_grd_noise 360 --train_grd_FOV $YOUR_FOV --test_grd_FOV $YOUR_FOV 
	CVACT: python train_cvact_fov.py --polar 1 --train_grd_noise 360 --train_grd_FOV $YOUR_FOV --test_grd_FOV $YOUR_FOV 

2. Evaluation:
	CVUSA: python test_cvusa_fov.py --polar 1 --train_grd_noise 360 --train_grd_FOV $YOUR_FOV --test_grd_FOV $YOUR_FOV 
	CVACT: python test_cvact_fov.py --polar 1 --train_grd_noise 360 --train_grd_FOV $YOUR_FOV --test_grd_FOV $YOUR_FOV 

Note that the test set construction operations are inside the data preparation script, polar_input_data_orien_FOV_3.py for CVUSA and ./OriNet_CVACT/input_data_act_polar_3.py for CVACT. We use "np.random.rand(2019)" in test_cvusa_fov.py and test_cvact_fov.py to make sure the constructed test set is the same one whenever they are used for performance evaluation for different models.

In case readers are interested to see the query images of newly constructed test sets where the ground images are with unkown orientation and small FoV, we provide the following two python scripts to save the images and their ground truth orientations at the local disk:

- CVUSA datset: python generate_test_data_cvusa.py 

- CVACT dataset: python generate_test_data_cvact.py

Readers are encouraged to visit "https://github.com/Liumouliu/OriCNN" to access codes for evaluation on the fine-grained geo-localization CVACT_test set. 

### Models:
Our trained models for CVUSA and CVACT are available in [here](https://drive.google.com/file/d/1jmBn6D6hfifKm6JYw0dc_WMRJ9bSLb5h/view?usp=sharing). 

There is also an "Initialize" model for your own training step. The VGG16 part in the "Initialize" model is initialised by the online model and other parts are initialised randomly. 

Please put them under the director of "Model/" and then you can use them for training or evaluation.



### Publications
This work is published in CVPR 2020.  
[Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching]

If you are interested in our work and use our code, we are pleased that you can cite the following publication:  
*Yujiao Shi, Xin Yu, Dylan Campbell, Hongdong Li. Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching.*

@inproceedings{shi2020where,
  title={Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching},
  author={Shi, Yujiao and Yu, Xin and Campbell, Dylan and Li, Hongdong},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}


