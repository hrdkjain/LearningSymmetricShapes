# Learning Symmetric Shapes

In this work, we propose an efficient iterative planar parameterization for disk topology shapes. The parameterization is used as a tool to regularize the mesh onto a square grid and encoded with vertex position. The resultant encoding is an image with rgb denoting xyz positions on the mesh.

The ICCV workshop paper can be found [here](https://openaccess.thecvf.com/content_ICCVW_2019/papers/GMDL/Jain_Learning_to_Reconstruct_Symmetric_Shapes_using_Planar_Parameterization_of_3D_ICCVW_2019_paper.pdf). First part of this project deals with parameterization and second deals with learning of shapes from geometry images. 

![teaser](Images/teaser.png)

## Installation
Code for Parameterization has been written in C++ and requires:
- CGAL [Fork with Iterative Parameterization Implementation](https://github.com/hrdkjain/cgal/tree/Iterative_authalic_parameterization)
- Boost
- OpenCV

Deep network code is based on Tensorflow and is tested on Ubuntu with:
- python (3.5.2)
- tensorflow-gpu (1.14)
- scikit-image (0.15.0)
- numpy (1.16.5)
- natsort
- tqdm

## Usage
### Part I: Parameterization
Code contains functionality for:
- slicing the mesh (--slice)
- Iterative Surface Parameterization with **_n_** iterations (--sPI **_n_**)
- Compute Geometry Image (of size **_im_**) from the parameterized representation (--m2G **_im_**)
- Remesh point cloud from Geometry Image (--G2o)

### Part II: Learning Shapes
python based functionality which contains:
- generating curvature mask from normalGI
- tensorflow model
- docker image
- python scripts to train and test the model

## Citation
If you find this project useful in your work, please consider citing:

    @inproceedings{jain2019learning,
      title={Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface},
      author={Jain, Hardik and Wöllhaf, Manuel and Hellwich, Olaf},
      booktitle={The IEEE International Conference on Computer Vision (ICCV) Workshops},
      year={2019}
    }

