/***************************************************************************************
 *    Title: Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface
 *    Conference: IEEE International Conference on Computer Vision (ICCV) Workshops
 *    Authors: Hardik Jain, Manuel Wöllhaf, Olaf Hellwich
 *    Date: 7 Oct. 2019
 *    Availability: https://github.com/hrdkjain/LearningSymmetricShapes
 *
 ***************************************************************************************/

#include "Parameterization.h"

Parameterization::Parameterization(std::stringstream & LogFile, fs::path inputPath, fs::path outputPath, bool &useNormal, int &im):
LogFile(LogFile), useNormal(useNormal), im_size(im)  {
  this->inputPath = inputPath;
  this->paramFile = (outputPath / inputPath.stem()).string() + "_arcSMI.off";

  // GI
  this->paramFile_flatGI = (paramFile.parent_path() / paramFile.stem()).string() + "_" + std::to_string(im_size) + "_flatGI.png";
  this->paramFile_nflatGI = (paramFile.parent_path() / paramFile.stem()).string() + "_" + std::to_string(im_size) + "_nflatGI.png";

  // GI2mesh
  this->paramFile_flatGI_off = (outputPath / inputPath.stem()).string() + ".off";
}

Parameterization::~Parameterization() {
  // TODO Auto-generated destructor stub
}

bool Parameterization::surfaceParameteriseIterative(int iterations) {
  // check if the parameterised file already exists
  if (outfileExists(paramFile, 10, ", surfParamed"))
    return true;

  // read input
  Surface_mesh sm;
  if(!meshLoader(inputPath, sm, " input mesh for parameterization", LogFile))
    return false;

  Border_parameterizer border_param;
  halfedge_descriptor bhd = CGAL::Polygon_mesh_processing::longest_border(sm).first;
  // The 2D points of the uv parametrisation will be written into this map
  UV_uhm uv_uhm;
  UV_pmap uv_map(uv_uhm);
  SMP::Error_code err;
  double error;
  try {
    err = SMP::parameterize(sm, Parameterizer(border_param), bhd, uv_map, iterations, error);
  } catch (...) {
    std::cerr << "  SMP::parameterize didn't succeed\n";
    LogFile << "SMP::parameterize didn't succeed\n";
    return false;
  }
  if (err != SMP::OK) {
    std::cerr << "  Error: " << SMP::get_error_message(err) << "\n";
    LogFile << SMP::get_error_message(err) << "\n";
    return false;
  }

  std::cout.width(4);
  std::cout << " " << iterations << std::flush;
  std::cout.width(10);
  std::cout << " " << error << std::flush;
  LogFile << iterations << "," << error << std::endl;

  std::ofstream out(paramFile.string().c_str());
  std::size_t vertices_counter = 0, faces_counter = 0;
  typedef boost::unordered_map<vertex_descriptor, std::size_t> Vertex_index_map;
  Vertex_index_map vium;
  boost::associative_property_map<Vertex_index_map> vimap(vium);
  if (out) {
    out << "OFF\n";
    out << sm.number_of_vertices() << " " << sm.number_of_faces() << " 0\n";
    boost::graph_traits<Surface_mesh>::vertex_iterator vit, vend;
    boost::tie(vit, vend) = vertices(sm);
    while (vit != vend) {
      vertex_descriptor vd = *vit++;
      out << get(uv_map, vd) << " 0\n";
      put(vimap, vd, vertices_counter++);
    }

    BOOST_FOREACH(face_descriptor fd, faces(sm)) {
      halfedge_descriptor hd = halfedge(fd, sm);
      out << "3";
      BOOST_FOREACH(vertex_descriptor vd, vertices_around_face(hd, sm)) {
        out << " " << get(vimap, vd);
      }
      out << '\n';
      faces_counter++;
    }
    if (vertices_counter != sm.number_of_vertices()) {
      std::cerr << "number of vertices in 3D 2D are not matching\n";
      LogFile << "number of vertices in 3D 2D are not matching\n";
      return false;
    } else if (faces_counter != sm.number_of_faces()) {
      std::cerr << "number of faces in 3D 2D are not matching\n";
      LogFile << "number of faces in 3D 2D are not matching\n";
      return false;
    }
    std::cout << ", surfParamed" << std::flush;
    return true;
  }
  else {
    std::cerr << "Could not open " << paramFile << " to write\n";
    LogFile << "Could not open " << paramFile << " to write\n";
    return false;
  }
  return true;
}

bool Parameterization::mesh2GI() {
  // check if the GI already exists
  if (outfileExists(paramFile_flatGI, 10, ", GIed")) {
    if (useNormal)  {
      if(outfileExists(paramFile_nflatGI, 10, ", normalGIed"))
        return true;
    }
    else
      return true;
  }

  Surface_mesh Mesh_3D;
  Surface_mesh Mesh_2D;

  // Start with Loading of Mesh_3D
  if (!meshLoader(inputPath, Mesh_3D, "3D mesh", LogFile, false))
    return false;

  // create the normal property map if normal GI needs to be computed
  Surface_mesh::Property_map<vertex_descriptor, Kernel::Vector_3> Mesh_3D_nm;
  if (useNormal) {
    Mesh_3D_nm = Mesh_3D.add_property_map<vertex_descriptor,Kernel::Vector_3>("v:normals", CGAL::NULL_VECTOR).first;
    PMP::compute_vertex_normals(Mesh_3D, Mesh_3D_nm);
  }

  if (!meshLoader(paramFile, Mesh_2D, "flat mesh", LogFile, false))
    return false;

  // check if Mesh_2D has any vertex which is an outlier
  BOOST_FOREACH(vertex_descriptor vd, vertices(Mesh_2D))  {
    Point_3 pt = Mesh_2D.point(vd);
    if(abs(pt.x()) > 1.5 || abs(pt.y()) > 1.5)  {
      std::cerr << " " << vd << "(" << pt.x() << "," << pt.y() << ")" << std::endl;
      LogFile << " " << vd << "(" << pt.x() << "," << pt.y() << ")" << std::endl;
      return false;
    }
  }

  // if the bvertexMap is false means there is no vertexMap
  // assert that the Mesh_2D and Mesh_3D have same number of vertices
  if(Mesh_2D.number_of_vertices() != Mesh_3D.number_of_vertices())  {
    std::cerr << "  Mesh_2D.nv != mesh_3D.nv" << std::endl;
    LogFile << "  Mesh_2D.nv != mesh_3D.nv" << std::endl;
    return false;
  }

  std::vector<vertex_descriptor> Mesh_3D_vds(Mesh_3D.num_vertices());
  vertex_iterator tmvi = CGAL::vertices(Mesh_3D).begin(), tmvi_end = CGAL::vertices(Mesh_3D).end();
  CGAL_For_all(tmvi, tmvi_end)
  Mesh_3D_vds.push_back(*tmvi);

  SM_vimap Mesh_2D_vimap = Mesh_2D.add_property_map<vertex_descriptor, int>("v:index").first;
  int i = 0;
  BOOST_FOREACH(vertex_descriptor vd, vertices(Mesh_2D))
  put(Mesh_2D_vimap, vd, i++);

  // matrices for GI
  cv::Mat M0 = cv::Mat::zeros(im_size, im_size, CV_32FC1);  // Geometry Image
  cv::Mat M1 = cv::Mat::zeros(im_size, im_size, CV_32FC1);  // Geometry Image
  cv::Mat M2 = cv::Mat::zeros(im_size, im_size, CV_32FC1);  // Geometry Image
  std::map<int, cv::Mat> outputMap;
  outputMap[0] = M0;
  outputMap[1] = M1;
  outputMap[2] = M2;

  // matrices for normal GI
  cv::Mat nM0 = cv::Mat::zeros(im_size, im_size, CV_32FC1);  // Geometry Image
  cv::Mat nM1 = cv::Mat::zeros(im_size, im_size, CV_32FC1);  // Geometry Image
  cv::Mat nM2 = cv::Mat::zeros(im_size, im_size, CV_32FC1);  // Geometry Image
  std::map<int, cv::Mat> noutputMap;
  noutputMap[0] = nM0;
  noutputMap[1] = nM1;
  noutputMap[2] = nM2;

  if (!useNormal) {
    nM0.release();
    nM1.release();
    nM2.release();
  }

  cv::Mat P = cv::Mat::zeros(2, 3, CV_32F); // Current 2D point
  cv::Mat V = cv::Mat::zeros(3, 3, CV_32F); // current dimension value of 3D Mesh
  cv::Mat nV = cv::Mat::zeros(3, 3, CV_32F);  // current dimension value of 3D Mesh Normals

  int f_idx = 0;
  cv::Mat Mnb = cv::Mat::zeros(im_size, im_size, CV_32FC1); // Geometry Image # pts calculator
  BOOST_FOREACH(face_descriptor fd, Mesh_2D.faces()) {
    int vt_count = 0;
    BOOST_FOREACH(vertex_descriptor vd_2D, vertices_around_face(Mesh_2D.halfedge(fd), Mesh_2D)) {
      P.at<float>(0, vt_count) = Mesh_2D.point(vd_2D)[0] * (im_size - 1);
      P.at<float>(1, vt_count) = Mesh_2D.point(vd_2D)[1] * (im_size - 1);
      // find the 3D mesh vertex index corresponding to the 2d mesh vd
      vertex_descriptor vd_3D;
      vd_3D = vd_2D;

      for (int dim = 0; dim < 3; ++dim) {
        //each Dimension of 3D mesh
        V.at<float>(dim, vt_count) = Mesh_3D.point(vd_3D)[dim];
        if (useNormal)
          nV.at<float>(dim, vt_count) = Mesh_3D_nm[vd_3D][dim];
      }
      vt_count++;
    }

    // V and P values corresponding to the current face is obtained
    // Now we work on these values
    cv::Mat pos, z1, z2;
    // Dimensions of the mesh grid
    int n_rows = Ceil(P.row(0)) - Floor(P.row(0)) + 1;
    int n_cols = Ceil(P.row(1)) - Floor(P.row(1)) + 1;
    if(n_rows<= 0 ||n_cols<= 0)
      continue;
    try {
      z1.create(n_rows, n_cols, CV_32FC1);
      z2.create(n_rows, n_cols, CV_32FC1);
    }
    catch(...)  {
      std::cerr << "Couldn't create z1 or z2\n";
      LogFile << "Couldn't create z1 or z2\n";
      return false;
    }
    for (int i = 0; i < z1.rows; ++i) {
      for (int j = 0; j < z1.cols; ++j) {
        z1.at<float>(i, j) = Floor(P.row(0)) + i;
        z2.at<float>(i, j) = Floor(P.row(1)) + j;
      }
    }

    z1 = z1.t();
    z1 = z1.reshape(0, 1);
    z2 = z2.t();
    z2 = z2.reshape(0, 1);
    cv::vconcat(z1, z2, pos);

    // barycentric coords computation
    cv::Mat tmp1, tmp2, c;
    cv::vconcat(cv::Mat::ones(1, P.cols, CV_32F), P, tmp1);
    cv::vconcat(cv::Mat::ones(1, pos.cols, CV_32F), pos, tmp2);
    cv::solve(tmp1, tmp2, c);

    // restrict the BC to inside triangle
    cv::Mat pos1, c1;
    c1.convertTo(c1, CV_32F);
    pos1.convertTo(pos1, CV_32F);
    for (int i = 0; i < c.cols; i++) {
      // cutoff was previously taken from octave but should be different for c++ beacuse of difference in datatypes
      if (c.at<float>(0, i) >= -0.000022204
          && c.at<float>(1, i) >= -0.000022204
          && c.at<float>(2, i) >= -0.000022204
          && !((c.at<float>(0, i) == 0) && (c.at<float>(1, i) == 0)
              && (c.at<float>(2, i) == 0))) {
        cv::Mat trash1 = c.col(i).t();
        c1.push_back(trash1);
        cv::Mat trash2 = pos.col(i).t();
        pos1.push_back(trash2);
      }
    }
    if (c1.empty() || pos1.empty()) {
      pos1.release();
      c1.release();
      f_idx++;
      continue;
    }
    c = c1.t();
    pos = pos1.t();
    pos1.release();
    c1.release();

    // restrict the pos to inside the image
    // TODO: this condition never seems to happen
    for (int i = 0; i < pos.cols; i++) {
      if (pos.at<float>(0, i) >= 0) {
        if (pos.at<float>(0, i) < im_size) {
          if (pos.at<float>(1, i) >= 0) {
            if (pos.at<float>(1, i) < im_size) {
              cv::Mat trash1 = c.col(i).t();
              c1.push_back(trash1);
              cv::Mat trash2 = pos.col(i).t();
              pos1.push_back(trash2);

            }
          }
        }
      }
    }

    if (c1.empty() || pos1.empty()) {
      pos1.release();
      c1.release();
      f_idx++;
      continue;
    }

    c = c1.t();
    pos = pos1.t();
    pos1.release();
    c1.release();

    // Now the value assignment has to be done
    for (int i = 0; i < pos.cols; i++) {
      int r_idx = (int) pos.at<float>(0, i);
      int c_idx = (int) pos.at<float>(1, i);
      for (int dim = 0; dim < 3; ++dim) {  //each Dimension of 3D mesh
        outputMap[dim].at<float>(r_idx, c_idx) =
            outputMap[dim].at<float>(r_idx, c_idx)
            + V.at<float>(dim, 0) * c.at<float>(0, i)
            + V.at<float>(dim, 1) * c.at<float>(1, i)
            + V.at<float>(dim, 2) * c.at<float>(2, i);
        if (useNormal)
          noutputMap[dim].at<float>(r_idx, c_idx) =
              noutputMap[dim].at<float>(r_idx, c_idx)
              + nV.at<float>(dim, 0) * c.at<float>(0, i)
              + nV.at<float>(dim, 1) * c.at<float>(1, i)
              + nV.at<float>(dim, 2) * c.at<float>(2, i);
      }
      Mnb.at<float>(r_idx, c_idx) = Mnb.at<float>(r_idx, c_idx) + 1;
    }
    f_idx++;
  } // for all faces

  // create masks from Mnb
  cv::Mat tmp_mask_value(im_size, im_size, CV_8UC1);
  cv::Mat tmp_mask_NaN(im_size, im_size, CV_8UC1);
  cv::threshold(Mnb, tmp_mask_value, 0, 255, cv::THRESH_BINARY);
  cv::threshold(Mnb, tmp_mask_NaN, 0, 255, cv::THRESH_BINARY_INV);

  cv::Mat mask_NaN, mask_value;
  tmp_mask_NaN.convertTo(mask_NaN, CV_8UC1);
  tmp_mask_value.convertTo(mask_value, CV_8UC1);
  tmp_mask_NaN.release();
  tmp_mask_value.release();

  // Specify convolution filter size required to compensate for no values in geometry image
  int filterX = 3;
  int filterY = 3;
  int nIter = 20;
  cv::Mat kernel = cv::Mat::ones(filterX, filterY, CV_32F) / (float) (filterX * filterY);

  for (int dim = 0; dim < 3; ++dim) {  //each Dimension of 3D mesh
    if (outputMap[dim].cols == 0) {
      LogFile << "outputMap is empty" << std::endl;
      std::cerr << "  OutputMap is empty" << std::endl;
      return false;
    }

    // divide by the number of vertices
    outputMap[dim] /= Mnb;
    noutputMap[dim] /= Mnb;
    // filter the NaNs
    // 1. for GI
    filterOutputMap(outputMap[dim], mask_value, mask_NaN, kernel, nIter);
    // 2. if required then for mormal GI
    if (useNormal)
      filterOutputMap(noutputMap[dim], mask_value, mask_NaN, kernel, nIter);
  } //each dimension

  // 1. for GI
  combineNSave(outputMap, paramFile_flatGI, ", savedGI");
  // 2. if required for Normal GI
  if(useNormal)
    combineNSave(noutputMap, paramFile_nflatGI, ", savednGI");

  return true;
}

bool Parameterization::GI2off()  {

  std::string inputPathStr = inputPath.string();
  if(inputPath.extension().string() == ".png") {
    // check input path and assign flatGI and nflatGI
    std::size_t GILoc = inputPathStr.find("nflatGI");
    if(GILoc!=std::string::npos)  {
      this->paramFile_nflatGI = inputPathStr;
      this->paramFile_flatGI = inputPathStr.substr(GILoc,6) + "flatGI" + inputPathStr.substr(GILoc+6);
    }
    else  {
      GILoc = inputPathStr.find("flatGI");
      if(GILoc!=std::string::npos)  {
        this->paramFile_flatGI = inputPathStr;
        this->paramFile_nflatGI = inputPathStr.substr(0,GILoc) + "nflatGI" + inputPathStr.substr(GILoc+6);
      }
      else  {
        this->paramFile_flatGI = inputPathStr;
      }
    }
  }

  if (outfileExists(paramFile_flatGI_off, 10, " , offED"))
    return true;

  // Get vertices
  if(!readGI())
    return false;

  // Get Surface Mesh from vertices
  Surface_mesh sm;
  if(!addVerticestoSM(sm))
    return false;

  if(!saveMesh(paramFile_flatGI_off, sm, ", off", LogFile))
    return false;

  return true;
}


//private
void Parameterization::filterOutputMap(cv::Mat&A, cv::Mat& mask_value, cv::Mat&mask_NaN,
    cv::Mat& kernel, int nIter) {
  // compute mean of value outputMap and fed those values to NaN
  float mean_outValue = cv::mean(A, mask_value)[0];
  cv::Mat out_tmp(A.size(), A.type(), cv::Scalar(mean_outValue));
  out_tmp.copyTo(A, mask_NaN);
  out_tmp.release();

  // Perform convolution using a ones filter
  cv::Mat outputMapTmp;
  A.copyTo(outputMapTmp);
  for (int i = 0; i < nIter; i++) {
    cv::filter2D(outputMapTmp, outputMapTmp, -1, kernel, cv::Point(-1, -1),
        0, cv::BORDER_DEFAULT);
    A.copyTo(outputMapTmp, mask_value);
  }
  outputMapTmp.copyTo(A);
  outputMapTmp.release();
}

void Parameterization::combineNSave(std::map<int, cv::Mat> &outMap, std::string meshFileFlatGI, std::string desc) {
  // this check is to ensure any of the previous files are not overwritten
  if(outfileExists(meshFileFlatGI, 10, desc+"ED"))
    return;

  // statistics for the three channels
  double minVal[3];
  double maxVal[3];
  // for each Dimension of 3D mesh
  for (int dim = 0; dim < 3; ++dim) {
    // find the min and max in each dimension/plane seperately
    cv::minMaxLoc(outMap[dim], &minVal[dim], &maxVal[dim], NULL, NULL);
    // subtract with the min so that the new min is Zero, equivalent to translation in 3D
    cv::subtract(outMap[dim], minVal[dim], outMap[dim]);
  }

  cv::Mat in[] = { outMap[2], outMap[1], outMap[0] };
  int from_to[] = { 0, 0, 1, 1, 2, 2 };
  cv::Mat M = cv::Mat::zeros(im_size, im_size, CV_32FC3); // 3D Geometry Image
  cv::mixChannels(in, 3, &M, 1, from_to, 3);

  // divide by maximum so that new maximum is one
  // equivalent to uniform scaling as it is applied to all the dimensions
  cv::divide(M, newMax(minVal, maxVal), M);

  cv::Mat MM(im_size, im_size, CV_8UC3);
  M.convertTo(MM, CV_8UC3, 255);
  // scale to 255 is required or else image wont be visible in other viewer

  // swap color channels because opencv saves as BGR
  cvtColor(MM, MM, cv::COLOR_RGB2BGR);
  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(0);

  cv::imwrite(meshFileFlatGI, MM, compression_params);
  std::cout << desc << std::flush;
}

double Parameterization::newMax(double minVal[3], double maxVal[3]) {
  double tmp, tmpVal[3];
  int J;
  for (int j = 0; j < 3; j++) {
    tmpVal[j] = maxVal[j] - minVal[j];
    if(true && j == 0)
      // if its a slice means that the x axis should be giving double length
      tmpVal[j] *= 2;
    if (j == 0) {
      tmp = tmpVal[j];
      J = 0;
    } else if (tmpVal[j] > tmp) {
      tmp = tmpVal[j];
      J = j;
    }
  }
  return tmpVal[J];
}

bool Parameterization::readGI(int downScaleFactor) {

  if (!infileExists(paramFile_flatGI, 0, "GI ", LogFile))
    return false;
  if (useNormal) {
    if (!infileExists(paramFile_nflatGI, 0, "Normal GI ", LogFile))
      return false;
  }

  cv::Mat Img;//(im_size, im_size, CV_32FC3);
  cv::Mat normalImg;//(im_size, im_size, CV_32FC3);
  if (!useNormal)
    normalImg.release();

  Img = cv::imread(paramFile_flatGI, cv::IMREAD_UNCHANGED);
  assert(downScaleFactor!= 0);
  if (downScaleFactor!= 1)
    cv::resize(Img, Img, cv::Size(Img.cols/downScaleFactor, Img.rows/downScaleFactor), 0, 0, CV_INTER_LINEAR);
  // as it is opencv read, image is read in BGR

  double minNormalImg, maxNormalImg;
  if (useNormal)  {
    normalImg = cv::imread(paramFile_nflatGI, cv::IMREAD_UNCHANGED); //, cv::IMREAD_ANYDEPTH); ////CV_LOAD_IMAGE_COLOR);
    if (downScaleFactor!= 1)
      cv::resize(normalImg, normalImg, cv::Size(normalImg.cols/downScaleFactor, normalImg.rows/downScaleFactor), 0, 0, CV_INTER_LINEAR);
    cv::minMaxLoc(normalImg, &minNormalImg, &maxNormalImg, NULL, NULL);
  }

  // find max of the GI image and divide by max to limit the range to 1
  double minImg, maxImg;
  cv::minMaxLoc(Img, &minImg, &maxImg, NULL, NULL);
  // however while choosing the maximum we must take care
  // as the GI from Mesher is 0-255
  maxImg = 255.0;

  int n_rows = Img.rows;
  int n_cols = Img.cols;

  // iterating for all the pixels of image
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      // as the image is read in B-G-R (0-1-2) assign accordingly to X-Y-Z
      float b = Img.at<cv::Vec3b>(i, j).val[0];
      float g = Img.at<cv::Vec3b>(i, j).val[1];
      float r = Img.at<cv::Vec3b>(i, j).val[2];
      // saving vertices
      verticesSS << "v " << (b-minImg)/maxImg << " " << (g-minImg)/maxImg << " " << (r-minImg)/maxImg << "\n";

      // saving normals
      if (useNormal) {
        float nb = normalImg.at<cv::Vec3b>(i, j).val[0];
        float ng = normalImg.at<cv::Vec3b>(i, j).val[1];
        float nr = normalImg.at<cv::Vec3b>(i, j).val[2];
        normalSS << "vn " << nb << " " << ng << " " << nr << "\n";
      }

      // saving faces
      // additional 1 added, as obj are 1 indexed
      if (i < n_rows - 1 && j < n_cols - 1) {
        faceSS << "f "
            << n_rows * (j) + i + 1 << " "
            << n_rows * (j) + i + 1 + 1 << " "
            << n_rows * (j + 1) + i + 1 + 1 << " "
            << n_rows * (j + 1) + i + 1 << "\n";
      }
    }
  }
  return true;
}

bool Parameterization::addVerticestoSM(Surface_mesh& sm)  {
  std::string temp;
  while(std::getline(verticesSS,temp,'\n')) {
    std::vector<std::string> pointVector = splitString(temp, " v");
    assert(pointVector.size()==3);
    Point_3 pt;
    try
    {
      pt = Point_3(
          boost::lexical_cast<double>(pointVector[0]),
          boost::lexical_cast<double>(pointVector[1]),
          boost::lexical_cast<double>(pointVector[2]));
    }
    catch (boost::bad_lexical_cast &)
    {
      std::cerr << "Unable to cast vertices to Point_3" << std::endl;
      LogFile << "Unable to cast vertices to Point_3" << std::endl;
      return false;
    }
    sm.add_vertex(pt);
  }
  return true;
}
