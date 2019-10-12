/***************************************************************************************
 *    Title: Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface
 *    Conference: IEEE International Conference on Computer Vision (ICCV) Workshops
 *    Authors: Hardik Jain, Manuel Wöllhaf, Olaf Hellwich
 *    Date: 7 Oct. 2019
 *    Availability: https://github.com/hrdkjain/LearningSymmetricShapes
 *
 ***************************************************************************************/

#ifndef MAIN_H_
#define MAIN_H_

#include "include.h"
#include "Texter.h"
#include "Preprocess.h"
#include "Parameterization.h"

bool getFlags (char * argv[], int argc);

flag Flag;
paths Paths;
std::ofstream LogFile;  // report log file
std::stringstream LogSS;  // log stringstream

#endif /* MAIN_H_ */
