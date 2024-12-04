#pragma once
#ifndef EINSUM_HPP
#define EINSUM_HPP
// Eigen3
#define EIGEN_USE_THREADS
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

// C++ std
#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// 定义宏来获取 OMP_NUM_THREADS，如果未设置，则使用本地 CPU 的线程数
#define GET_OMP_NUM_THREADS(n_thread)                             \
    do {                                                          \
        const char* omp_env_var = std::getenv("OMP_NUM_THREADS"); \
        if (omp_env_var) {                                        \
            n_thread = std::atoi(omp_env_var);                    \
        }                                                         \
        else {                                                    \
            n_thread = std::thread::hardware_concurrency();       \
        }                                                         \
    } while (0)

namespace YXTensor {
// main functions
template <typename TensorType>
bool tensor_equal(const TensorType& tensor1, const TensorType& tensor2,
    const double tol = 1e-10);

template <typename TensorType>
void print_tensor(const TensorType& tensor);

template <typename TensorType>
Eigen::Tensor<TensorType, 2> matrix_to_tensor(const Eigen::Matrix<TensorType, Eigen::Dynamic, Eigen::Dynamic>& matrix);

template <typename TensorType>
Eigen::Matrix<TensorType, Eigen::Dynamic, Eigen::Dynamic> tensor_to_matrix(const Eigen::Tensor<TensorType, 2>& tensor);

template <int num_contractions, typename TensorType, int Dim1, int Dim2,
    int ResultDim>
Eigen::Tensor<TensorType, ResultDim>
einsum(std::string&& einsum_str, const Eigen::Tensor<TensorType, Dim1>&& input1,
    const Eigen::Tensor<TensorType, Dim2>&& input2);

template <int num_contractions, typename TensorType, int Dim1, int Dim2,
    int ResultDim>
void einsum(const std::string&& einsum_str,
    const Eigen::Tensor<TensorType, Dim1>&& input1,
    const Eigen::Tensor<TensorType, Dim2>&& input2,
    Eigen::Tensor<TensorType, ResultDim>&& result_input);

// namespace Tensor

// Print formatted values
/**
 * 打印格式化的值。
 *
 * 这个模板函数根据传入的值的类型进行格式化打印。如果是整数类型，它将打印一个宽度为
 * 12 的零填充整数。如果是浮点类型，它将打印一个宽度为 0，精度为 10
 * 的浮点数格式。
 *
 * @tparam T 要打印的值的类型，可以是整数类型或浮点类型。
 * @param value 要打印的值。
 */
template <typename T>
void print_formatted(const T& value)
{
    if constexpr (std::is_integral_v<T>) {
        // fmt::print("{:12}", value); // Zero-padded integers
        std::cout << std::format("{: 12}", value);
    }
    else if constexpr (std::is_floating_point_v<T>) {
        // fmt::print("{:0.10f}", value);
        std::cout << std::format("{: 0.10f}", value);
    }
}

// Recursive tensor printing
/**
 * 递归打印张量的内容
 *
 * @param tensor 要打印的张量对象
 * @param shape 张量的形状，一个整数向量，表示每个维度的大小
 * @param dim 当前递归到的维度
 * @param indices 当前维度的索引向量
 * @param indent 缩进字符串，用于表示递归层次
 * @param is_last 是否是当前维度的最后一个元素
 */
template <typename TensorType>
void print_recursive(const TensorType& tensor,
    const typename TensorType::Dimensions& shape, size_t dim,
    std::vector<size_t> indices, const std::string& indent,
    bool is_last)
{
    if (dim == shape.size()) {
        Eigen::array<Eigen::Index, TensorType::NumDimensions> eigen_indices;
        std::copy(indices.begin(), indices.end(), eigen_indices.begin());
        print_formatted(tensor(eigen_indices));
    }
    else {
        // fmt::print("[");
        std::cout << "[";
        std::string new_indent = indent + " ";
        for (size_t i = 0; i < shape[dim]; ++i) {
            indices.emplace_back(i);
            print_recursive(tensor, shape, dim + 1, indices, new_indent,
                i == shape[dim] - 1);
            indices.pop_back();
            if (i < shape[dim] - 1) {
                // fmt::print(" ");
                std::cout << " ";
            }
        }
        // fmt::print("]");
        std::cout << "]";
        if (!is_last) {
            // fmt::print("\n{}", indent);
            std::cout << std::format("\n{}", indent);
        }
    }
}

/**
 * 打印张量的内容
 *
 * 这个模板函数递归地打印张量的内容。它首先打印一个空格，然后调用`print_recursive`函数来打印张量的内容。`print_recursive`函数根据张量的维度和当前维度的索引，递归地打印张量中的每个元素。最后，`print_tensor`函数在打印完张量内容后，打印一个换行符。
 *
 * @tparam TensorType 张量的类型，需要支持.dimensions()成员函数来获取维度信息
 * @param tensor 要打印的张量对象
 */
template <typename TensorType>
void print_tensor(const TensorType& tensor)
{
    const auto& shape = tensor.dimensions();
    // fmt::print(" ");
    std::cout << " ";
    print_recursive(tensor, shape, 0, std::vector<size_t>(), "", true);
    // fmt::print("\n");
    std::cout << std::endl;
}

// Check if tensors are equal
/**
 * 检查两个张量是否相等
 *
 * 这个函数将检查两个张量的维度是否匹配，以及它们的数据是否在给定的容差范围内相等。如果维度不匹配，或者任何对应位置的数据点差异大于容差，函数将返回
 * false；如果所有数据点的差异都在容差范围内，函数将返回 true。
 *
 * @tparam TensorType
 * 张量的类型，需要支持.dimensions()成员函数来获取维度信息，以及.data()成员函数来获取数据指针
 * @param tensor1 第一个要比较的张量对象
 * @param tensor2 第二个要比较的张量对象
 * @param tol 容差，用来确定两个元素是否被认为是相等的。默认值为 1e-10
 * @return 如果两个张量的所有元素在容差范围内相等，则为 true；否则为 false
 */
template <typename TensorType>
bool tensor_equal(const TensorType& tensor1, const TensorType& tensor2,
    const double tol)
{
    if (tensor1.dimensions() != tensor2.dimensions()) {
        return false;
    }
    for (auto i = 0; i < tensor1.size(); ++i) {
        if (std::abs(tensor1.data()[i] - tensor2.data()[i]) > tol) {
            return false;
        }
    }
    return true;
}

template <typename TensorType>
Eigen::Tensor<TensorType, 2> matrix_to_tensor(const Eigen::Matrix<TensorType, Eigen::Dynamic, Eigen::Dynamic>& matrix)
{
    // 获取矩阵的尺寸
    int rows = matrix.rows();
    int cols = matrix.cols();
    Eigen::Tensor<TensorType, 2> tensor(rows, cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            tensor(i, j) = matrix(i, j);
        }
    }

    return tensor;
}

template <typename TensorType>
Eigen::Matrix<TensorType, Eigen::Dynamic, Eigen::Dynamic> tensor_to_matrix(const Eigen::Tensor<TensorType, 2>& tensor)
{
    int rows = tensor.dimension(0);
    int cols = tensor.dimension(1);
    Eigen::Matrix<TensorType, Eigen::Dynamic, Eigen::Dynamic> matrix(rows, cols);

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            matrix(i, j) = tensor(i, j);
        }
    }

    return matrix;
}
// Parse einsum string and perform contractions
/**
 * 解析爱因斯坦求和约定字符串
 *
 * 这个函数解析一个爱因斯坦求和约定字符串，并将结果存储在几个向量和一个字符串中，这些字符串和向量后来被用于进一步的计算。
 *
 * @param einsum_str 要解析的爱因斯坦求和约定字符串。这个字符串应该包含箭头 "->"
 * 来分隔输入部分和输出部分。
 * @param result_indices
 * 输出字符串，包含爱因斯坦求和约定字符串中的索引，这些索引确定输出张量的维度。
 * @param shuffle_indexs
 * 索引的向量，这些索引表明结果应该如何在不同维度上进行重新排列。
 * @param left
 * 索引的向量，这些索引指向输入字符串的差异元素在左边字符串中的位置。
 * @param right
 * 索引的向量，这些索引指向输入字符串的差异元素在右边字符串中的位置。
 * @return 一个
 * std::vector<Eigen::IndexPair<int>>，其中每个元素都是一个包含两个索引的对；这些索引对对应于解析字符串时找到的匹配索引。
 * @throws std::invalid_argument 如果爱因斯坦求和约定字符串格式不正确，例如缺少
 * "->"，或者如果左边和右边字符串的长度不同，或者如果找不到输出索引，函数将抛出
 * std::invalid_argument 异常。
 *
 */
template <typename TensorType>
std::vector<Eigen::IndexPair<int>>
parse_einsum_string(const std::string& einsum_str, std::string& result_indices,
    std::vector<size_t>& shuffle_indexs,
    std::vector<size_t>& left, std::vector<size_t>& right)
{
    std::vector<Eigen::IndexPair<int>> idx_pairs;

    auto arrow_pos = einsum_str.find("->");
    if (arrow_pos == std::string::npos) {
        throw std::invalid_argument("Invalid einsum string format: missing '->'");
    }
    auto left_part = einsum_str.substr(0, arrow_pos);
    result_indices = einsum_str.substr(arrow_pos + 2);

    if (result_indices.empty()) {
        throw std::invalid_argument(
            "Unsupport this code syntax, please set an index for output");
    }

    std::vector<std::string> input_parts;
    auto comma_pos = left_part.find(',');
    while (comma_pos != std::string::npos) {
        input_parts.emplace_back(left_part.substr(0, comma_pos));
        left_part = left_part.substr(comma_pos + 1);
        comma_pos = left_part.find(',');
    }
    input_parts.emplace_back(left_part);

    if (input_parts.size() != 2) {
        throw std::invalid_argument(
            "Invalid number of input tensors in einsum string");
    }

    const auto& I_indices = input_parts[0];
    const auto& D_indices = input_parts[1];

    auto find_different_element =
        [](const std::string& str1, const std::string& str2,
            std::vector<size_t>& left, std::vector<size_t>& right) -> std::string {
        std::vector<char> difference;
        for (size_t i = 0; i < str1.size(); ++i) {
            char c = str1[i];
            if (str2.find(c) == std::string::npos) {
                difference.emplace_back(c);
                left.emplace_back(i);
            }
        }
        for (size_t i = 0; i < str2.size(); ++i) {
            char c = str2[i];
            if (str1.find(c) == std::string::npos) {
                difference.emplace_back(c);
                right.emplace_back(i);
            }
        }
        return std::string(difference.begin(), difference.end());
    };

    auto find_indices =
        [](const std::string& result_indices,
            const std::string& different_elements) -> std::vector<size_t> {
        std::vector<size_t> indices;
        if (result_indices == different_elements) {
            return indices;
        }
        for (char c : different_elements) {
            auto pos = result_indices.find(c);
            if (pos != std::string::npos) {
                indices.emplace_back(pos);
            }
        }
        return indices;
    };

    auto different_elements = find_different_element(I_indices, D_indices, left, right);
    shuffle_indexs = find_indices(result_indices, different_elements);

    for (int i = 0; i < I_indices.size(); ++i) {
        for (int j = 0; j < D_indices.size(); ++j) {
            if (I_indices[i] == D_indices[j]) {
                idx_pairs.emplace_back(i, j);
            }
        }
    }

    return idx_pairs;
}

/**
 * 使用爱因斯坦求和约定（Einstein Summation
 * Convention）执行张量收缩操作，并在需要时对结果进行重排序。
 *
 * @tparam TensorType 表示张量的数据类型。
 * @tparam ResultDim 表示结果张量的维度。
 * @param einsum_str 一个字符串，指定爱因斯坦求和约定表示的索引操作。
 * @param input1 第一个输入张量。
 * @param input2 第二个输入张量。
 * @return 收缩操作的结果作为一个 Eigen::Tensor<TensorType, ResultDim>
 * 类型的张量。
 * @throws std::invalid_argument 如果 einsum_str
 * 格式不正确，或者输入张量的维度与 einsum_str
 * 指定的索引不匹配，或者收缩维度错误。
 */
template <int num_contractions, typename TensorType, int Dim1, int Dim2,
    int ResultDim>
Eigen::Tensor<TensorType, ResultDim>
einsum(std::string&& einsum_str, const Eigen::Tensor<TensorType, Dim1>&& input1,
    const Eigen::Tensor<TensorType, Dim2>&& input2)
{

    std::vector<size_t> left_idx;
    std::vector<size_t> right_idx;
    std::vector<size_t> shuffle_idx;
    std::string result_indices;

    auto idx_pairs = parse_einsum_string<TensorType>(
        einsum_str, result_indices, shuffle_idx, left_idx, right_idx);

    if (num_contractions != idx_pairs.size()) {
        throw std::invalid_argument("Invalid number of contractions");
    }

    Eigen::array<Eigen::IndexPair<int>, num_contractions> contract_dims;
    std::copy(idx_pairs.begin(), idx_pairs.end(), contract_dims.begin());
    Eigen::Tensor<TensorType, ResultDim> result;
    Eigen::array<Eigen::Index, ResultDim> result_dimensions;

    if (ResultDim != left_idx.size() + right_idx.size()) {
        throw std::invalid_argument("Invalid number of dimensions in result");
    }

    std::transform(left_idx.begin(), left_idx.end(), result_dimensions.begin(),
        [&](size_t idx) { return input1.dimension(idx); });

    std::transform(right_idx.begin(), right_idx.end(),
        result_dimensions.begin() + left_idx.size(),
        [&](size_t idx) { return input2.dimension(idx); });
    result.resize(result_dimensions);

    int n_thread;
    GET_OMP_NUM_THREADS(n_thread);
    std::cout << "Number of threads to use: " << n_thread << std::endl;

    Eigen::ThreadPool pool(n_thread);
    Eigen::ThreadPoolDevice my_device(&pool, n_thread);
    result.device(my_device) = std::move(input1).contract(std::move(input2), contract_dims);

    if (shuffle_idx.empty()) {
        return result;
    }
    else {
        Eigen::array<int, ResultDim> shuffle_array;
        std::copy(shuffle_idx.begin(), shuffle_idx.end(), shuffle_array.begin());
        return result.shuffle(shuffle_array);
    }
}
template <int num_contractions, typename TensorType, int Dim1, int Dim2,
    int ResultDim>
void einsum(const std::string&& einsum_str,
    const Eigen::Tensor<TensorType, Dim1>&& input1,
    const Eigen::Tensor<TensorType, Dim2>&& input2,
    Eigen::Tensor<TensorType, ResultDim>&& result_input)
{
    std::vector<size_t> left_idx;
    std::vector<size_t> right_idx;
    std::vector<size_t> shuffle_idx;
    std::string result_indices;

    auto idx_pairs = parse_einsum_string<TensorType>(
        einsum_str, result_indices, shuffle_idx, left_idx, right_idx);

    if (num_contractions != idx_pairs.size()) {
        throw std::invalid_argument("Invalid number of contractions");
    }

    Eigen::array<Eigen::IndexPair<int>, num_contractions> contract_dims;
    std::copy(idx_pairs.begin(), idx_pairs.end(), contract_dims.begin());
    Eigen::Tensor<TensorType, ResultDim> result;
    Eigen::array<Eigen::Index, ResultDim> result_dimensions;

    if (ResultDim != left_idx.size() + right_idx.size()) {
        throw std::invalid_argument("Invalid number of dimensions in result");
    }

    std::transform(left_idx.begin(), left_idx.end(), result_dimensions.begin(),
        [&](size_t idx) { return input1.dimension(idx); });

    std::transform(right_idx.begin(), right_idx.end(),
        result_dimensions.begin() + left_idx.size(),
        [&](size_t idx) { return input2.dimension(idx); });
    result.resize(result_dimensions);

    int n_thread;
    GET_OMP_NUM_THREADS(n_thread);
    std::cout << "Number of threads to use: " << n_thread << std::endl;

    Eigen::ThreadPool pool(n_thread);
    Eigen::ThreadPoolDevice my_device(&pool, n_thread);
    result.device(my_device) = std::move(input1).contract(std::move(input2), contract_dims);

    if (shuffle_idx.empty()) {
        result_input = result;
    }
    else {
        Eigen::array<int, ResultDim> shuffle_array;
        std::copy(shuffle_idx.begin(), shuffle_idx.end(), shuffle_array.begin());
        result_input = result.shuffle(shuffle_array);
    }
}

} // namespace YXTensor
#endif
