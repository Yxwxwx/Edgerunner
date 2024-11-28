#pragma once
#include "gto/gto.hpp" // 确保正确包含 GTO 命名空间
#include <Eigen/Dense>
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>

namespace Integral {
template <typename Scalar> class Integral {
private:
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> S;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> V;
  Eigen::Tensor<Scalar, 4> I;

public:
  Integral(GTO::Mol mol);
  void calculate_1e();
  void calculate_2e();
};
} // namespace Integral