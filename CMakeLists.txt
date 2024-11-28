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

# Head Only Third Party
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/json)

# Libcint 
# --------------------> libcint (required) <--------------------
ExternalProject_Add(
  cint
  PREFIX ${CMAKE_BINARY_DIR}
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/libcint
  INSTALL_DIR ${CMAKE_BINARY_DIR}/libcint
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DENABLE_STATIC=ON
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/libcint
)
ExternalProject_Get_Property(cint BINARY_DIR)
get_filename_component(CINTROOT ${BINARY_DIR} ABSOLUTE)
include_directories(${CINTROOT}/include)
set(CINTLIB "${CINTROOT}/libcint.a")  


find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

file(GLOB_RECURSE src_sto "src/gto/*.cpp")
file(GLOB_RECURSE src_sto_int "src/integral/*.cpp")
file(GLOB_RECURSE src_main "src/hf/*.cpp")

add_definitions(-DSHARE_DIR="${CMAKE_SOURCE_DIR}/share")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(Edgerunners ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${src_sto} ${src_sto_int})

target_link_libraries(Edgerunners ${CINTLIB} OpenMP::OpenMP_CXX Threads::Threads)