#pragma once

#include "har/tensor.hpp"

#include <concepts>
#include <stdexcept>
#include <type_traits>

namespace har::math {

// Matrix multiplication: C = A @ B
// A: [M, K], B: [K, N] -> C: [M, N]
template <std::floating_point T>
auto matmul(const Tensor<T> &a, const Tensor<T> &b) -> Tensor<T> {
  if (a.dims() != 2 || b.dims() != 2) {
    throw std::invalid_argument("matmul requires 2D tensors");
  }

  const size_t M = a.shape()[0];
  const size_t K = a.shape()[1];
  const size_t N = b.shape()[1];

  if (b.shape()[0] != K) {
    throw std::invalid_argument("matmul dimension mismatch");
  }

  Tensor<T> c({M, N}, T{0}, a.device());

#ifdef HAR_HAS_CUDA
  if constexpr (std::is_same_v<T, float>) {
    if (a.device() == Device::CUDA && cuda_ops::active()) {
      cuda_ops::matmul(a.data(), b.data(), c.data(), static_cast<int>(M),
                       static_cast<int>(K), static_cast<int>(N));
      return c;
    }
  }
#endif

  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      const T a_ik = a.at(i, k);
      for (size_t j = 0; j < N; ++j) {
        c.at(i, j) += a_ik * b.at(k, j);
      }
    }
  }

  return c;
}

template <std::floating_point T>
auto dot(const Tensor<T> &a, const Tensor<T> &b) -> T {
  if (a.dims() != 1 || b.dims() != 1 || a.size() != b.size()) {
    throw std::invalid_argument("dot requires 1D tensors of same size");
  }

  T result{0};
  a.sync();
  b.sync();
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

// Outer product: a ⊗ b
// a: [M], b: [N] -> [M, N]
template <std::floating_point T>
auto outer(const Tensor<T> &a, const Tensor<T> &b) -> Tensor<T> {
  if (a.dims() != 1 || b.dims() != 1) {
    throw std::invalid_argument("outer requires 1D tensors");
  }

  const size_t M = a.size();
  const size_t N = b.size();
  Tensor<T> result({M, N}, a.device());
  a.sync();
  b.sync();

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      result.at(i, j) = a[i] * b[j];
    }
  }

  return result;
}

template <std::floating_point T>
auto transpose(const Tensor<T> &a) -> Tensor<T> {
  if (a.dims() != 2) {
    throw std::invalid_argument("transpose requires 2D tensor");
  }

  const size_t rows = a.shape()[0];
  const size_t cols = a.shape()[1];
  Tensor<T> result({cols, rows}, a.device());

#ifdef HAR_HAS_CUDA
  if constexpr (std::is_same_v<T, float>) {
    if (a.device() == Device::CUDA && cuda_ops::active()) {
      cuda_ops::transpose(a.data(), result.data(), static_cast<int>(rows),
                          static_cast<int>(cols));
      return result;
    }
  }
#endif

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result.at(j, i) = a.at(i, j);
    }
  }

  return result;
}

} // namespace har::math
