/***************************************************************************************
 *    Title: Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface
 *    Conference: IEEE International Conference on Computer Vision (ICCV) Workshops
 *    Authors: Hardik Jain, Manuel Wöllhaf, Olaf Hellwich
 *    Date: 7 Oct. 2019
 *    Availability: https://github.com/hrdkjain/LearningSymmetricShapes
 *
 ***************************************************************************************/

#ifndef PARAMETERIZATION_H_
#define PARAMETERIZATION_H_

#include "include.h"

#include <CGAL/Unique_hash_map.h>
typedef CGAL::Unique_hash_map<vertex_descriptor, Point_2> UV_uhm;
typedef boost::associative_property_map<UV_uhm> UV_pmap;

#include <CGAL/Surface_mesh_parameterization/Square_border_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Iterative_parameterize.h>
#include <CGAL/Surface_mesh_parameterization/Iterative_authalic_parameterizer_3.h>
namespace SMP = CGAL::Surface_mesh_parameterization;
typedef SMP::Square_border_arc_length_parameterizer_3<Surface_mesh> Border_parameterizer;
typedef SMP::Iterative_authalic_parameterizer_3<Surface_mesh, Border_parameterizer> Parameterizer;
namespace PMP = CGAL::Polygon_mesh_processing;
#include <CGAL/Polygon_mesh_processing/compute_normal.h>

class Parameterization {
public:
  Parameterization(std::stringstream & LogFile, fs::path inputPath, fs::path outputPath, bool &useNormal, int &im);
  virtual ~Parameterization();
  bool surfaceParameteriseIterative(int iterations);
  bool mesh2GI();
  bool GI2off();


private:
  void filterOutputMap(cv::Mat&A, cv::Mat& mask_value, cv::Mat&mask_NaN, cv::Mat& kernel, int nIter);
  void combineNSave(std::map<int, cv::Mat> &outMap, std::string meshFileFlatGI, std::string desc);
  double newMax(double minVal[3], double maxVal[3]);
  bool readGI(int downScaleFactor=1);
  bool addVerticestoSM(Surface_mesh& sm);

  std::stringstream& LogFile;
  fs::path inputPath; // input path
  bool useNormal; // use normals with geometry image
  int im_size;

  fs::path paramFile; // surface paramterized mesh
  std::string paramFile_flatGI; // vertex encoded geometry image
  std::string paramFile_nflatGI;  // normal encoded geometry image
  fs::path paramFile_flatGI_off; // remeshed pointcloud

  std::stringstream verticesSS, normalSS, faceSS; //strings to read mesh from GI
};

#endif /* PARAMETERIZATION_H_ */
