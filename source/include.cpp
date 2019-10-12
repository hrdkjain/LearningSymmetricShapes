/***************************************************************************************
 *    Title: Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface
 *    Conference: IEEE International Conference on Computer Vision (ICCV) Workshops
 *    Authors: Hardik Jain, Manuel Wöllhaf, Olaf Hellwich
 *    Date: 7 Oct. 2019
 *    Availability: https://github.com/hrdkjain/LearningSymmetricShapes
 *
 ***************************************************************************************/

#include "include.h"

bool outfileExists(fs::path outFilePath, const int size, std::string printDesc) {
  if (fs::exists(outFilePath)) {
    if (fs::file_size(outFilePath) > size) {
      std::cout << printDesc << std::flush;
      return true;
    } else
      return false;
  } else
    return false;
}

bool infileExists(fs::path inFilePath, const int size, std::string errDesc, std::stringstream& LogFile) {
  if (fs::exists(inFilePath)) {
    //check for size of file
    if (fs::file_size(inFilePath) <= size) {
      std::cerr << "\t Empty" << errDesc << "\n";
      LogFile << "Empty" << errDesc << "\n";
      return false;
    }
    else
      return true;
  } else {
    std::cerr << "\t No " << errDesc << "\n";
    LogFile << "No " << errDesc << "\n";
    return false;
  }
}

bool meshLoader(fs::path meshFile, Surface_mesh& loadedMesh, std::string fileDesc, std::stringstream& LogFile,
    bool bdebug) {
  // check if the file exists
  if (fs::exists(meshFile) && fs::is_regular_file(meshFile)) {
    fs::ifstream in_fs(meshFile);
    if (!in_fs) {
      std::cerr << "\t Unable to create fs for " << fileDesc << std::endl;
      LogFile << "Unable to create fs for " << fileDesc << "\n";
      return false;
    }

    try {
      if (!(in_fs >> loadedMesh)) {
        std::cerr << std::setw(20) << "\t Unable to read " << fileDesc << std::endl;
        LogFile << "Unable to read " << fileDesc << "\n";
        return false;
      }
    } catch (...) {
      std::cerr << std::setw(20) << "\t Unable to read " << fileDesc << std::endl;
      LogFile << "Unable to read " << fileDesc << "\n";
      return false;
    }

    if (loadedMesh.number_of_vertices() == 0) {
      std::cerr << "\t" << fileDesc << "has no vertices" << std::endl;
      LogFile << "\t" << fileDesc << "has no vertices" << std::endl;
      return false;
    } else {
      if (bdebug)
        std::cout << "Loaded Mesh " << meshFile << " has " << loadedMesh.number_of_vertices() << " Vertices "
        << std::flush;
    }
    in_fs.close();
    return true;
  } else {
    std::cerr << "\t" << fileDesc << "doesn't exists\n";
    LogFile << fileDesc << "doesn't exists\n";
    return false;
  }
}

bool MLS(fs::path inputPath,fs::path outputPath, std::string mlxScript, std::string desc, std::stringstream& LogFile, std::string option) {
  std::string bash = "meshlabserver -i "+inputPath.string()+" -o "+outputPath.string();
  if (option!="NULL")
    bash += " "+ option;
  bash += " -s ";

  bash += mlxScript;
  bash += ".mlx";
  int i = system(bash.c_str());
  if(i != 0)  {
    std::cerr << "\tCouldn't process " << desc << " in meshlab\n";
    LogFile << "Couldn't process " << desc << " in meshlab\n";
    return false;
  }
  return true;
}

int Ceil(cv::Mat A) {
  int tmp = ceil(A.at<float>(0, 0));
  for (int i = 1; i < A.cols; ++i) {
    tmp = ceil(A.at<float>(0, i)) > tmp ? ceil(A.at<float>(0, i)) : tmp;
  }
  return tmp;
}

int Floor(cv::Mat A) {
  int tmp = floor(A.at<float>(0, 0));
  for (int i = 1; i < A.cols; ++i) {
    tmp = floor(A.at<float>(0, i)) < tmp ? floor(A.at<float>(0, i)) : tmp;
  }
  return tmp;
}

bool saveMesh(fs::path meshFile, Surface_mesh& sm, std::string fileDesc, std::stringstream& LogFile)  {
  fs::ofstream out_fs(meshFile);
  out_fs << sm;
  out_fs.close();
  std::cout << fileDesc << std::flush;
  return true;
}

std::vector<std::string> splitString(std::string str, std::string delimiters) {
  std::vector<std::string> parts;
  boost::split(parts, str, boost::is_any_of(delimiters));

  std::vector<std::string> newParts;
  for (std::vector<std::string>::iterator it = parts.begin(); it != parts.end(); it++)  {
    if (!(*it).empty())
      newParts.push_back(*it);
  }
  return newParts;
}
