/***************************************************************************************
 *    Title: Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface
 *    Conference: IEEE International Conference on Computer Vision (ICCV) Workshops
 *    Authors: Hardik Jain, Manuel Wöllhaf, Olaf Hellwich
 *    Date: 7 Oct. 2019
 *    Availability: https://github.com/hrdkjain/LearningSymmetricShapes
 *
 ***************************************************************************************/

#ifndef INCLUDE_H_
#define INCLUDE_H_

// std inlcudes
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <random>
#include <signal.h>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <vector>

// Boost includes
#include <boost/filesystem.hpp>
namespace fs = ::boost::filesystem;
#include <boost/filesystem/fstream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

// CGAL Includes
// Kernels
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_2 Point_2;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Plane_3 K_Plane_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
typedef boost::graph_traits<Surface_mesh>::vertex_descriptor vertex_descriptor;
typedef boost::graph_traits<Surface_mesh>::edge_descriptor edge_descriptor;
typedef boost::graph_traits<Surface_mesh>::halfedge_descriptor halfedge_descriptor;
typedef boost::graph_traits<Surface_mesh>::face_descriptor face_descriptor;
typedef boost::graph_traits<Surface_mesh>::vertex_iterator vertex_iterator;
typedef Surface_mesh::Property_map<vertex_descriptor, int> SM_vimap;

// openCV Includes
#include <opencv2/opencv.hpp>

struct paths  {
  fs::path listFilePath;  // file path of the list file
  fs::path DBPath;  // parent path of the list file
  std::string fldPre; // folder which is used for listing file as well as to output folder
  std::string ext;  // extension of files to be listed
  std::vector<fs::path> modelFilePathList;  // vector containing list of files to be processed
  std::string flStr;  // file string requried for selective listing of files
};

struct flag {
  bool slice; // slice input mesh
  bool sPI; // iterative surface parameterization
  bool m2G; // parameterized mesh to geometry image
  bool G2o; // Geometry image to remesh point cloud
  bool useNormal; // use normals for geometry image or remesh generation
  int sPIterations; // maximum number of iterations of surface parameterization
  int im_size;  // size of geometry image
};

bool outfileExists(fs::path outFilePath, const int size, std::string printDesc);
bool infileExists(fs::path inFilePath, const int size, std::string errDesc, std::stringstream& LogFile);
bool meshLoader(fs::path meshFile, Surface_mesh& loadedMesh, std::string fileDesc, std::stringstream& LogFile, bool bdebug=false);
bool MLS(fs::path inputPath,fs::path outputPath, std::string mlxScript, std::string desc, std::stringstream& LogFile, std::string option="NULL");
int Ceil(cv::Mat A);
int Floor(cv::Mat A);
bool saveMesh(fs::path meshFile, Surface_mesh& sm, std::string fileDesc, std::stringstream& LogFile);
std::vector<std::string> splitString(std::string str, std::string delimiters);

#endif /* INCLUDE_H_ */
