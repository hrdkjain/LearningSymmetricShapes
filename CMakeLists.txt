PROJECT(LearningSymmetricShapes)

cmake_minimum_required(VERSION 2.6.2)
if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" VERSION_GREATER 2.6)
  if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" VERSION_GREATER 2.8.3)
    cmake_policy(VERSION 2.8.4)
  else()
    cmake_policy(VERSION 2.6)
  endif()
endif()
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Modules/" ${CMAKE_MODULE_PATH})
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)  
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -std=c++11")

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Boost and its components
find_package(Boost REQUIRED)
if(NOT Boost_FOUND)
  message(STATUS "This project requires the Boost library, and will not be compiled.")
  return()  
endif()

# CGAL and its components
find_package(CGAL REQUIRED)
if(NOT CGAL_FOUND)
  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()  
endif()
include(${CGAL_USE_FILE})
find_package(GMP REQUIRED ./Find)
if(NOT GMP_FOUND)
  message(STATUS "This project requires GMP and is not found, and will not be compiled.")
  return()  
endif()
include_directories(${GMP_INCLUDE_DIR})
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)

set(CMAKE_BUILD_TYPE Release)

file(GLOB SOURCE_FILES source/*.cpp source/*.h)
ADD_EXECUTABLE(Main ${SOURCE_FILES})
#TARGET_LINK_LIBRARIES(Main ${OpenCV_LIBS} -lboost_filesystem -lboost_system -lCGAL ${CGAL_3RD_PARTY_LIBRARIES} -lmpfr -lgmpxx -lgmp -lgsl -lm)
TARGET_LINK_LIBRARIES(Main ${OpenCV_LIBS} ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES} ${GMP_LIBRARIES} ${PCL_LIBRARIES} ${CPPL_LIBS} ${SMI_LIBS} ${PYTHON_LIBRARIES} ${GSL_LIBRARIES} -lCGAL -lm -lmpfr -lboost_filesystem -lboost_system -lgmpxx -lgmp)# -lgsl -lgslcblas)
# add the install targets 
install (TARGETS Main DESTINATION ~/bin)
