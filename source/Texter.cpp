/***************************************************************************************
 *    Title: Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface
 *    Conference: IEEE International Conference on Computer Vision (ICCV) Workshops
 *    Authors: Hardik Jain, Manuel Wöllhaf, Olaf Hellwich
 *    Date: 7 Oct. 2019
 *    Availability: https://github.com/hrdkjain/LearningSymmetricShapes
 *
 ***************************************************************************************/

#include "Texter.h"

Texter::Texter(paths& Paths) {
  this->Paths = &Paths;
}

bool Texter::listFilesToFile()  {
  // list files from the specified folder to a list file
  if(Paths->fldPre == "NULL")
    Paths->fldPre = Paths->listFilePath.stem().string() + "/";

  // List files from folder
  std::cout << "Listing all files" << " with extension " << Paths->ext << " from folder " << Paths->fldPre << std::flush;
  if(Paths->flStr != "NULL")
    std::cout << " with flStr " << Paths->flStr << std::flush;
  getFiles(Paths->DBPath/Paths->fldPre, Paths->modelFilePathList);
  std::cout << ", found " << Paths->modelFilePathList.size() << " elements"<< std::endl;

  // Write the list to file
  if (Paths->modelFilePathList.size() == 0) {
    std::cerr << "Empty list\n";
    return false;
  }
  std::cout << "Writing to " << Paths->listFilePath << std::endl;
  fs::ofstream listFilePath_fs;
  listFilePath_fs.open(Paths->listFilePath);
  for (std::vector<fs::path>::iterator i = Paths->modelFilePathList.begin(); i != Paths->modelFilePathList.end(); ++i)  {
    listFilePath_fs << i->string() << "\n";
  }
  listFilePath_fs.close();
  return true;
}

bool Texter::listFilesFromFile()  {
  // list files from the list file for processing
  std::cout << "Processing all files from list file " << Paths->listFilePath<< std::flush;

  fs::ifstream listFilePath_fs(Paths->listFilePath);
  if (listFilePath_fs.fail()) {
    std::cerr << "Couldn't open List file to read\n";
    throw std::ios_base::failure(std::strerror(errno));
    return false;
  }
  if (Paths->listFilePath.extension().string() == ".txt")  {
    while (!listFilePath_fs.eof()) {
      std::string line;
      getline(listFilePath_fs, line);
      if (line.empty())
        continue;
      else if (line.substr(0, 1) == "#")
        continue;
      else if (line.substr(0, 3) == "EOF")
        break;
      Paths->modelFilePathList.push_back(line);
    }
    listFilePath_fs.close();
  }
  else {
    std::cerr << "un-specified listFilePath Extension" << std::endl;
    return false;
  }

  std::cout << ", found " << Paths->modelFilePathList.size() << " elements"<< std::endl;
  if(Paths->modelFilePathList.size() == 0)
    return false;
  return true;
}

Texter::~Texter() {
  // TODO Auto-generated destructor stub
}

void Texter::getFiles(fs::path root, std::vector<fs::path>& ret) {
  // return the path of all files that have the specified extension
  std::vector<std::string> vStr;
  for (fs::recursive_directory_iterator it(root); it != fs::recursive_directory_iterator(); ++it) {
    vStr.push_back(it->path().string());
  }
  std::sort(vStr.begin(), vStr.end(), natural_less<std::string>);
  std::vector<fs::path> v(vStr.begin(), vStr.end());

  // filter file list for specified extension and flStr
  for (std::vector<fs::path>::iterator it(v.begin()), it_end(v.end()); it != it_end; ++it) {
    if (fs::is_regular_file(*it) && it->extension() == Paths->ext) {
      if((Paths->flStr != "NULL") && (it->stem().string().find(Paths->flStr)==std::string::npos))
        continue;
      ret.push_back(*it);
    }
  }
}
