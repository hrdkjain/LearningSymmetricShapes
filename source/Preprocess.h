/***************************************************************************************
 *    Title: Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface
 *    Conference: IEEE International Conference on Computer Vision (ICCV) Workshops
 *    Authors: Hardik Jain, Manuel Wöllhaf, Olaf Hellwich
 *    Date: 7 Oct. 2019
 *    Availability: https://github.com/hrdkjain/LearningSymmetricShapes
 *
 ***************************************************************************************/

#ifndef PREPROCESS_H_
#define PREPROCESS_H_

#include "include.h"

typedef CGAL::Aff_transformation_3<Kernel> K_AffineTran;
#include <CGAL/Polygon_mesh_processing/distance.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/refine.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
namespace PMP = CGAL::Polygon_mesh_processing;


class Preprocess {
public:
  Preprocess(std::stringstream & LogFile, fs::path inputPath, fs::path outputPath);
  virtual ~Preprocess();
  bool slice();


private:
  bool saveSlice(fs::path& filepath, Surface_mesh &sm);
  void refineOnly(Surface_mesh &sm);
  bool closeHoles(fs::path& filepath);

  fs::path inputPath, outputPath;
  bool bdebug;
  std::stringstream& LogFile;
};

#endif /* PREPROCESS_H_ */
