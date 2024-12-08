# Edgerunner

This is a programming example for quantum chemistry education, written in C++ and supporting OpenMP parallelization.The name is inspired by the anime *Cyberpunk: Edgerunners*.

## Directory Structure

**CMakeLists.txt**: Managed by CMake, automatically installs the Libcint integral library, Eigen3 linear algebra library, etc.

**share/basis**: Stores basis set information, including Pople series, def2 series, and Dunning's related consistent (cc) basis sets.

**src**: Source code files.

- **src/gto**: Builds molecular models, reads basis set information, and stores it according to Libcint's requirements.
- **src/integral**: Computes electron integrals.
- **src/hf**: Hartree-Fock method.
- **src/mp2**: Integral transformations and MP2 method.
- **src/linalg**: Some algorithms, currently only the einsum method for tensor contraction based on einsum.

## Features

- Supports single-electron and two-electron integral calculations in spherical harmonic coordinates (sph), with the ability to specify shells for two-electron integrals.
- Supports Restricted Hartree-Fock (**RHF**) calculations, with **CDIIS** method and **Direct-SCF** support.
- Supports integral transformations from Atomic Orbitals (AO) to Molecular Orbitals (MO).
- Supports energy calculation using the **MP2** method.
- Supports **einsum** computation between two tensors of arbitrary order.
- All methods support **OpenMP/thread-based** parallelism.

## Installation

- Requires **CMake** version >= 3.28, **Ninja** is recommended as the generator.
- A C/C++ compiler supporting **C++20** and above (Clang/GCC/Intel_oneapi) is required.
- Supported only on **MacOS** and **Linux** systems.

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -G "Ninja" ..
(if you use MKL as BLAS and LAPACK, add -D__USE_MKL__=ON)
cmake --build .
```

## Usage

As a framework for teaching and exercises, you can easily run different systems by modifying the `main` function:

1. Include necessary headers:

```cpp
#include "gto/gto.hpp"
#include "hf/hf.hpp"
#include "mp2/mp2.hpp"
```

2. Construct the molecule:

```cpp
GTO::Mol H2O("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "cc-pvqz");
```

- XYZ coordinates (case-insensitive, atoms separated by `;`)
- Basis set name (case-insensitive, but symbols like `-`, `*`must be correct)
- 2S(NOT 2S + 1)
- Charge

3. Initialize the method:

```cpp
HF::RHF hf(H2O);  // RHF method
MP2::MP2 mp2(H2O);  // MP2 method
```

4. Run the calculation:

```cpp
[Method].kernel();
```
