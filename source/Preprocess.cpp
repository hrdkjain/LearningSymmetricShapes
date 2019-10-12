/***************************************************************************************
 *    Title: Learning to Reconstruct Symmetric Shapes using Planar Parameterization of 3D Surface
 *    Conference: IEEE International Conference on Computer Vision (ICCV) Workshops
 *    Authors: Hardik Jain, Manuel Wöllhaf, Olaf Hellwich
 *    Date: 7 Oct. 2019
 *    Availability: https://github.com/hrdkjain/LearningSymmetricShapes
 *
 ***************************************************************************************/

#include "Preprocess.h"

Preprocess::Preprocess(std::stringstream & LogFile, fs::path inputPath, fs::path outputPath): LogFile(LogFile) {
  this->inputPath = inputPath;
  this->outputPath = (outputPath / inputPath.stem()).string() + ".off";
  this->bdebug = false;
}

Preprocess::~Preprocess() {
  // TODO Auto-generated destructor stub
}

bool Preprocess::slice()  {
  // check if output file already exists
  if(outfileExists(outputPath, 10, " ,sliced"))
    return true;

  // read input
  Surface_mesh inMesh;
  if(!meshLoader(inputPath, inMesh, " input mesh for slicing", LogFile, bdebug))
    return false;

  CGAL::Bbox_3 bbox = PMP::bbox(inMesh);
  double slicePlane;
  slicePlane = (bbox.xmin()+bbox.xmax())/2;
  K_Plane_3 plane(1,0,0,-slicePlane);

  // Check the direction of the plane and orient it to the negative x axis
  // such that the slice is present on the positive x axis
  Kernel::Direction_3 dir = plane.orthogonal_direction();
  if(dir.dx()>0)
    plane = plane.opposite();

  // clip the mesh with the plane
  try{
    PMP::clip(inMesh, plane);
  }
  catch(...)  {
    std::cerr << "unable to slice mesh\n";
    LogFile << "unable to slice mesh\n";
    return false;
  }

  // also do the translation in x so that the new xMin is zero
  // this is done later as the close holes might add some vertices near to the
  Kernel::Vector_3 K_translateVector(Point_3(slicePlane,0.0,0.0),Point_3(0.0,0.0,0.0));
  K_AffineTran t(CGAL::TRANSLATION,K_translateVector);
  BOOST_FOREACH(vertex_descriptor vd, inMesh.vertices())  {
    inMesh.point(vd) = inMesh.point(vd).transform(t);
  }

  // save the slice while closing holes
  if(!saveSlice(outputPath, inMesh))
    return false;
  std::cout << ", slice" << std::flush;
  return true;
}


// private
bool Preprocess::saveSlice(fs::path & filepath, Surface_mesh &sm) {
  // redundant cleaning steps are required to avoid any holes or non-manifoldness in the output

  // 1. CGAL based refining
  refineOnly(sm);
  if(!saveMesh(filepath, sm, ", refined mesh", LogFile))
    return false;

  // 2. Meshlab based non-manifold removal
  if(!MLS(filepath, filepath, "source/cleanSlice", " slice cleaning", LogFile))
    return false;

  // 3. CGAL based hole closing, for holes created by non-manifoldness removal
  if(!closeHoles(filepath))
    return false;

  return true;
}

void Preprocess::refineOnly(Surface_mesh &sm) {
  // connected comp
  PMP::keep_largest_connected_components(sm, 1, CGAL::parameters::all_default());

  //refine
  std::vector<vertex_descriptor> newVertices;
  std::vector<face_descriptor> newFaces;
  PMP::refine(sm, faces(sm),
      std::back_inserter(newFaces),
      std::back_inserter(newVertices));
}

bool Preprocess::closeHoles(fs::path& filepath)  {
  // read input
  Surface_mesh sm;
  if(!meshLoader(filepath, sm, " input mesh for closeASML", LogFile, bdebug))
    return false;

  // identify all the border vertices
  std::vector<halfedge_descriptor> bHalfEdges;
  PMP::border_halfedges(faces(sm), sm, std::back_inserter(bHalfEdges));

  // identify the longest border
  std::vector<halfedge_descriptor> bLHalfEdges;
  halfedge_descriptor bhd = PMP::longest_border(sm).first;
  BOOST_FOREACH(halfedge_descriptor bh, halfedges_around_face(bhd, sm)) {
    bLHalfEdges.push_back(bh);
  }

  int bCounter = bHalfEdges.size();
  int bLCounter = bLHalfEdges.size();

  if(bCounter > bLCounter)  {
    // meaning that there are some vertices lying on border other than the largest border
    // for those half-edges fill the holes
    std::vector<halfedge_descriptor> bnLHalfEdges(bCounter-bLCounter);
    BOOST_FOREACH(halfedge_descriptor h, halfedges(sm)) {
      std::vector<halfedge_descriptor>::iterator vecLoc = std::find(bLHalfEdges.begin(), bLHalfEdges.end(), h);
      if(is_border(h, sm)  && vecLoc == bLHalfEdges.end()) {
        std::vector<face_descriptor>  patch_facets;
        std::vector<vertex_descriptor>  patch_vertices;
        try {
          PMP::triangulate_hole(sm,h,std::back_inserter(patch_facets)
          ,PMP::parameters::vertex_point_map(get(CGAL::vertex_point, sm)).geom_traits(Kernel()));
        }
        catch(...)  {
          continue;
        }
      }
    }

  }
  // something is wrong
  else if(bCounter < bLCounter)  {
    return false;
  }

  if(!saveMesh(filepath, sm, ", hole closed", LogFile))
    return false;

  return true;
}
