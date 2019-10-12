/***************************************************************************************
 *    Title: Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface
 *    Conference: IEEE International Conference on Computer Vision (ICCV) Workshops
 *    Authors: Hardik Jain, Manuel Wöllhaf, Olaf Hellwich
 *    Date: 7 Oct. 2019
 *    Availability: https://github.com/hrdkjain/LearningSymmetricShapes
 *
 ***************************************************************************************/

#include "Main.h"

int main(int argc, char * argv[]) {
  std::chrono::high_resolution_clock::time_point begin_main = std::chrono::high_resolution_clock::now();
  int pr_mode = (argc > 1) ? atoi(argv[1]) : 0;
  Paths.listFilePath = (argc > 2) ? argv[2] : "";
  Paths.ext = ".off";
  Paths.fldPre = "NULL";
  Paths.flStr = "NULL";
  Paths.DBPath = Paths.listFilePath.parent_path();

  if(!getFlags(argv, argc))
    return -1;

  // Texter: Handles saving and reading of file lists based on pr_mode
  Texter Tx(Paths);
  if (pr_mode == 0) {
    if(!Tx.listFilesToFile())
      return -1;
    return 0;
  }
  else if (pr_mode == 1)	{
    if(!Tx.listFilesFromFile())
      return -1;
  }

  // Report Log File
  std::string logFilePath = (Paths.DBPath / ("Report_" + Paths.listFilePath.stem().string() + "_")).string();
  if (Paths.fldPre != std::string("NULL"))
    logFilePath += Paths.fldPre.substr(0, Paths.fldPre.length() - 1);
  else {
    srand(time(NULL));
    logFilePath += std::to_string(rand());
  }
  logFilePath += ".txt";
  LogFile.open(logFilePath.c_str(), std::ios::app);
  if (LogFile.fail()) {
    std::cerr << "    Couldn't open LogFile to write\n";
    throw std::ios_base::failure(std::strerror(errno));
    return -1;
  }
  else  {
    std::time_t timeStamp = std::time(nullptr);
    std::cout << std::ctime(&timeStamp);
    LogFile << std::ctime(&timeStamp);
    LogFile << "Processing Files from list file " << Paths.listFilePath << " & writing files to " << Paths.fldPre << std::endl;
  }

  fs::path outModelFilePath = Paths.DBPath / Paths.fldPre;
  // create the required paths so as to arrange the output systematically
  if (outModelFilePath.string()[outModelFilePath.string().size()-1] != '/')
    outModelFilePath = outModelFilePath.string() + "/";
  std::size_t backslash = outModelFilePath.string().find('/');
  while (backslash != std::string::npos) {
    fs::create_directory(outModelFilePath.string().substr(0, backslash + 1));
    backslash = outModelFilePath.string().find('/', backslash + 1);
  }

  int counter = 0;
  // execute main for list of all files in modelFilePathList
  for (int i = 0; i < Paths.modelFilePathList.size(); i++) {
    std::chrono::high_resolution_clock::time_point begin_t = std::chrono::high_resolution_clock::now();
    if (!LogSS.str().empty()) {
      LogFile << LogSS.str() << std::flush;
      LogSS.str(std::string());
    }
    fs::path modelFilePath = Paths.modelFilePathList[i];
    LogSS << modelFilePath.string() << " : " << std::flush;
    std::cout << modelFilePath.string() << " : " << std::flush;

    // Preprocess: Slice input mesh for parameterization
    Preprocess PP(LogSS, modelFilePath, outModelFilePath);
    if (Flag.slice) {
      if(!PP.slice())
        continue;
    }

    // Parameterization: To perform iterative parameterization, obtain geometry image and remesh from GI
    Parameterization PM(LogSS, modelFilePath, outModelFilePath, Flag.useNormal, Flag.im_size);
    if (Flag.sPI)  {
      if(!PM.surfaceParameteriseIterative(Flag.sPIterations))
        continue;
    }
    if (Flag.m2G)  {
      if(!PM.mesh2GI())
        continue;
    }
    else if (Flag.G2o) {
      if(!PM.GI2off())
        continue;
    }

    std::time_t timeStamp = std::time(nullptr);
    std::stringstream tmpSS;
    tmpSS << " ******************** " << ++counter << "-" << i + 1 << "/" << Paths.modelFilePathList.size() << " ("
        << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now()-begin_t).count()
        << " sec) " << std::ctime(&timeStamp);
    std::cout << tmpSS.str();
    LogSS << tmpSS.str();
  }
  // required for last file
  if (!LogSS.str().empty())
    LogFile << LogSS.str();

  std::stringstream tmpSS;
  tmpSS << "Finished in "<< std::chrono::duration_cast<std::chrono::minutes>(std::chrono::high_resolution_clock::now()-begin_main).count() << " min " << std::endl;
  std::cout << tmpSS.str();
  LogFile << tmpSS.str();

  LogFile.close();
  std::cout << "Log written to " << logFilePath << std::endl;
  return 0;
}

bool getFlags(char * argv[], int argc) {
  std::vector<std::string> args(argv, argv + argc);
  for (int i = 3; i < args.size(); ++i) {
    if (argv[i] == std::string("--ext"))
      Paths.ext = argv[++i];
    else if (argv[i] == std::string("--fldPre"))
      Paths.fldPre = argv[++i];
    else if (argv[i] == std::string("--flStr"))
      Paths.flStr = argv[++i];
    else if (argv[i] == std::string("--slice"))
      Flag.slice = true;
    else if (argv[i] == std::string("--sPI")) {
      Flag.sPI = true;
      Flag.sPIterations = atoi(argv[++i]);
    }
    else if (argv[i] == std::string("--m2G")) {
      Flag.m2G = true;
      Flag.im_size = atoi(argv[++i]);
    }
    else if (argv[i] == std::string("--useNormal"))
      Flag.useNormal = true;
    else if (argv[i] == std::string("--G2o"))
      Flag.G2o = true;
    else  {
      std::cerr << "Flag: " << argv[i] << " not defined in program\n";
      return false;
    }
  }
  return true;
}
