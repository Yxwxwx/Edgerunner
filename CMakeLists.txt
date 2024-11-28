cmake_minimum_required(VERSION 3.28)

project(Edgerunners LANGUAGES C CXX)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(UNIX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  endif(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
endif()

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


include(FetchContent)
# --------------------> json (required) <--------------------
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# --------------------> libcint (required) <--------------------
include(FetchContent)

FetchContent_Declare(
  cint
  GIT_REPOSITORY https://github.com/sunqm/libcint.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(cint)
include_directories(${cint_BINARY_DIR}/include)
set(CINTLIB "${cint_BINARY_DIR}/libcint.so")  # 如果你使用的是 .so 动态库



find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

file(GLOB_RECURSE src_gto "src/gto/*.cpp")
file(GLOB_RECURSE src_int "src/integral/*.cpp")
file(GLOB_RECURSE src_main "src/hf/*.cpp")

add_definitions(-DSHARE_DIR="${CMAKE_SOURCE_DIR}/share")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(Edgerunners ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${src_gto} ${src_int})

target_link_libraries(Edgerunners ${CINTLIB} OpenMP::OpenMP_CXX Threads::Threads  nlohmann_json::nlohmann_json)