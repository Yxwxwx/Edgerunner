cmake_minimum_required(VERSION 3.28)

project(Edgerunners LANGUAGES C CXX)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param=destructive-interference-size=64")

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

# 添加 CMake 配置选项
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
# 设置 CMake 编译选项
include_directories(${cint_BINARY_DIR}/include)
set(CINTLIB "${cint_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}cint${CMAKE_SHARED_LIBRARY_SUFFIX}")



find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

file(GLOB_RECURSE src_gto "src/gto/*.[ch]pp")
file(GLOB_RECURSE src_int "src/integral/*.[ch]pp")
file(GLOB_RECURSE src_hf "src/hf/*.[ch]pp")
file(GLOB_RECURSE src_util "src/linalg/*.[ch]pp")

add_definitions(-DSHARE_DIR="${CMAKE_SOURCE_DIR}/share")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(Edgerunners ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${src_gto} ${src_int} ${src_hf} ${src_util})

target_link_libraries(Edgerunners 
                      ${CINTLIB}
                      OpenMP::OpenMP_CXX 
                      Threads::Threads  )

set_target_properties(Edgerunners PROPERTIES
  BUILD_RPATH "${cint_BINARY_DIR}"
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

add_custom_command(TARGET Edgerunners POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_SOURCE_DIR}/share
      ${CMAKE_BINARY_DIR}/share
)

