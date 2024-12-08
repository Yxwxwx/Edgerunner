cmake_minimum_required(VERSION 3.28)

# --------------------> Intel oneAPI (opentional) <--------------------
option(USE_INTEL "Enable Intel oneAPI support" OFF)

if(USE_INTEL)
  message(STATUS "Intel oneAPI support enabled.")
  set(CMAKE_C_COMPILER "icx")
  set(CMAKE_CXX_COMPILER "icpx")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ipo -xHost")
else()
  message(STATUS "Intel oneAPI support not enabled.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()
# --------------------> compiler flags <--------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 --param=destructive-interference-size=64")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")

project(Edgerunner LANGUAGES C CXX)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# --------------------> eigen3 (required) <--------------------
include(FetchContent)

FetchContent_Declare(
  eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(eigen3)
include_directories(${eigen3_SOURCE_DIR})

# --------------------> libcint (required) <--------------------
FetchContent_Declare(
  cint
  GIT_REPOSITORY https://github.com/sunqm/libcint.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)
set(CMAKE_ARGS
  -DWITH_RANGE_COULOMB=ON
  -DENABLE_EXAMPLE=0
  -DENABLE_TEST=0
  -DWITH_F12=1
  -DWITH_COULOMB_ERF=1
  -DMIN_EXPCUTOFF=20
  -DKEEP_GOING=1
)
FetchContent_MakeAvailable(cint)
include_directories(${cint_BINARY_DIR}/include)
set(CINTLIB "${cint_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}cint${CMAKE_SHARED_LIBRARY_SUFFIX}")
# --------------------> MKL (required) <--------------------

option(USE_MKL "Enable MKL support" OFF)

if(USE_MKL)
    message(STATUS "MKL support enabled.")
    if(USE_INTEL) 
      set(MKLLIB "-qmkl=parallel")
    else()
      set(MKLLIB "-L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
      include_directories(${MKLROOT}/include)
    endif()
else()
    message(STATUS "MKL support disabled.")
    set(MKLLIB "")
endif()
# --------------------> END  <--------------------

file(GLOB_RECURSE src_gto "src/gto/*.[ch]pp")
file(GLOB_RECURSE src_int "src/integral/*.[ch]pp")
file(GLOB_RECURSE src_hf "src/hf/*.[ch]pp")
file(GLOB_RECURSE src_util "src/linalg/*.[ch]pp")
file(GLOB_RECURSE src_mp2 "src/mp2/*.[ch]pp")

add_definitions(-DSHARE_DIR="${CMAKE_SOURCE_DIR}/share")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(Edgerunner ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${src_gto} ${src_int} ${src_hf} ${src_util} ${src_mp2})

target_link_libraries(Edgerunner PRIVATE
                      ${CINTLIB}
                      ${MKLLIB}
                      OpenMP::OpenMP_CXX 
                      Threads::Threads  )

set_target_properties(Edgerunner PROPERTIES
  BUILD_RPATH "${cint_BINARY_DIR}"
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

add_custom_command(TARGET Edgerunner POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_SOURCE_DIR}/share
      ${CMAKE_BINARY_DIR}/share
)

