# Learning Symmetric Shapes
This project is based on the workshop paper ["Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface"](https://www.researchgate.net/publication/335528865_Learning_to_Reconstruct_Symmetric_Shapes_using_Planar_Parameterization_of_3D_Surface). First part of this project deals with parameterization and second deals with learning of shapes from geometry images. 

## Citation
If you find this project useful in your work, please consider citing:

    @inproceedings{jain2019learning,
      title={Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface},
      author={Jain, Hardik and Wöllhaf, Manuel and Hellwich, Olaf},
      booktitle={The IEEE International Conference on Computer Vision (ICCV) Workshops},
      year={2019}
    }

## Installation
Code for Parameterization has been written in C++ and requires:
- CGAL [Fork with Iterative Parameterization Implementation](https://github.com/hrdkjain/cgal/tree/Iterative_authalic_parameterization)
- Boost
- OpenCV

Deep network code is written in Tensorflow (1.12).

## Usage
### Part I: Parameterization
Code contains functionality for:
- slicing the mesh (--slice)
- Iterative Surface Parameterization with **_n_** iterations (--sPI **_n_**)
- Compute Geometry Image (of size **_im_**) from the parameterized representation (--m2G **_im_**)
- Remesh point cloud from Geometry Image (--G2o)

### Part II: Learning Shapes


## License
