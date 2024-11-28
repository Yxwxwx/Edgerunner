cmake_minimum_required(VERSION 3.28)

project(Edgerunners LANGUAGES C CXX)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
# --------------------> json (required) <--------------------
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# --------------------> libcint (required) <--------------------
FetchContent_Declare(
  cint
  GIT_REPOSITORY https://github.com/sunqm/libcint.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(cint)
include_directories(${cint_BINARY_DIR}/include)
set(CINTLIB "${cint_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}cint${CMAKE_SHARED_LIBRARY_SUFFIX}")



find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

file(GLOB_RECURSE src_gto "src/gto/*.[ch]pp")
file(GLOB_RECURSE src_int "src/integral/*.[ch]pp")
file(GLOB_RECURSE src_main "src/hf/*.[ch]pp")
file(GLOB_RECURSE src_util "src/linalg/*.[ch]pp")

add_definitions(-DSHARE_DIR="${CMAKE_SOURCE_DIR}/share")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(Edgerunners ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${src_gto} ${src_int})

target_link_libraries(Edgerunners 
                      ${CINTLIB}
                      OpenMP::OpenMP_CXX 
                      Threads::Threads  
                      nlohmann_json::nlohmann_json)

add_custom_command(TARGET Edgerunners POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_SOURCE_DIR}/share
      ${CMAKE_BINARY_DIR}/share
)