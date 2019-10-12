/***************************************************************************************
 *    Title: Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface
 *    Conference: IEEE International Conference on Computer Vision (ICCV) Workshops
 *    Authors: Hardik Jain, Manuel Wöllhaf, Olaf Hellwich
 *    Date: 7 Oct. 2019
 *    Availability: https://github.com/hrdkjain/LearningSymmetricShapes
 *
 ***************************************************************************************/

#ifndef TEXTER_H_
#define TEXTER_H_

#include "include.h"
#include "naturalorder.h"

class Texter {
public:
  Texter(paths & Paths);
  virtual ~Texter();
  bool listFilesFromFile();
  bool listFilesToFile();


private:
  void getFiles(fs::path root, std::vector<fs::path>& ret);

  paths * Paths;
};

#endif /* TEXTER_H_ */
