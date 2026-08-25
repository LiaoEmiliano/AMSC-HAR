#pragma once

#include "har/layers/layer.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace har::layers {

// 1D convolution over the time axis of per-frame features.
// input:  [N*T, C_in]  (frames laid out as n0t0, n0t1, ..., n1t0, ...)
// weight: [C_out, C_in, K]
// output: [N*T, C_out]
template <std::floating_point T = float> class TemporalConv1D {
public:
  TemporalConv1D(size_t in_channels, size_t out_channels, size_t kernel_size = 3,
                 bool bias = true)
      : in_channels_(in_channels), out_channels_(out_channels),
        kernel_size_(kernel_size), pad_(kernel_size / 2), use_bias_(bias),
        weight_({out_channels, in_channels, kernel_size}),
        bias_(bias ? Parameter<T>({out_channels}, T{0}) : Parameter<T>{}) {
    if (kernel_size == 0 || (kernel_size % 2) == 0) {
      throw std::invalid_argument(
          "TemporalConv1D: kernel_size must be a positive odd integer");
    }
    init_weights();
  }

  auto forward(const Tensor<T> &input, size_t n, size_t t) -> Tensor<T> {
    if (input.dims() != 2 || input.shape()[0] != n * t ||
        input.shape()[1] != in_channels_) {
      throw std::invalid_argument("TemporalConv1D: expected [N*T, C_in]");
    }
    input_cache_ = input;
    n_ = n;
    t_ = t;
    Tensor<T> out({n * t, out_channels_}, T{0}, input.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::temporal_conv1d_forward(
            input.data(), weight_.data.data(),
            use_bias_ ? bias_.data.data() : nullptr, out.data(),
            static_cast<int>(n), static_cast<int>(t),
            static_cast<int>(in_channels_), static_cast<int>(out_channels_),
            static_cast<int>(kernel_size_), static_cast<int>(pad_),
            use_bias_ ? 1 : 0);
        return out;
      }
    }
#endif

    input.sync();
    weight_.data.sync();
    if (use_bias_) {
      bias_.data.sync();
    }
    for (size_t ni = 0; ni < n; ++ni) {
      for (size_t ti = 0; ti < t; ++ti) {
        for (size_t co = 0; co < out_channels_; ++co) {
          T sum = use_bias_ ? bias_.data[co] : T{0};
          for (size_t k = 0; k < kernel_size_; ++k) {
            const size_t src_t = clamp_t(static_cast<int>(ti + k) - static_cast<int>(pad_),
                                         t);
            for (size_t ci = 0; ci < in_channels_; ++ci) {
              sum += weight_.data.at(co, ci, k) *
                     input.at(ni * t + src_t, ci);
            }
          }
          out.at(ni * t + ti, co) = sum;
        }
      }
    }
    out.sync();
    return out;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> {
    if (grad_output.dims() != 2 || grad_output.shape()[0] != n_ * t_ ||
        grad_output.shape()[1] != out_channels_) {
      throw std::invalid_argument("TemporalConv1D: grad shape mismatch");
    }
    Tensor<T> grad_input({n_ * t_, in_channels_}, T{0}, grad_output.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (grad_output.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::temporal_conv1d_backward(
            input_cache_.data(), weight_.data.data(), grad_output.data(),
            grad_input.data(), weight_.grad.data(),
            use_bias_ ? bias_.grad.data() : nullptr, static_cast<int>(n_),
            static_cast<int>(t_), static_cast<int>(in_channels_),
            static_cast<int>(out_channels_), static_cast<int>(kernel_size_),
            static_cast<int>(pad_), use_bias_ ? 1 : 0);
        return grad_input;
      }
    }
#endif

    input_cache_.sync();
    weight_.data.sync();
    grad_output.sync();
    weight_.grad.sync();
    if (use_bias_) {
      bias_.grad.sync();
    }

    for (size_t ni = 0; ni < n_; ++ni) {
      for (size_t ti = 0; ti < t_; ++ti) {
        for (size_t co = 0; co < out_channels_; ++co) {
          const T go = grad_output.at(ni * t_ + ti, co);
          if (use_bias_) {
            bias_.grad[co] += go;
          }
          for (size_t k = 0; k < kernel_size_; ++k) {
            const size_t src_t = clamp_t(static_cast<int>(ti + k) - static_cast<int>(pad_),
                                         t_);
            for (size_t ci = 0; ci < in_channels_; ++ci) {
              const T x = input_cache_.at(ni * t_ + src_t, ci);
              weight_.grad.at(co, ci, k) += go * x;
              grad_input.at(ni * t_ + src_t, ci) +=
                  go * weight_.data.at(co, ci, k);
            }
          }
        }
      }
    }
    weight_.grad.sync();
    if (use_bias_) {
      bias_.grad.sync();
    }
    grad_input.sync();
    return grad_input;
  }

  auto parameters() -> std::vector<Parameter<T> *> {
    if (use_bias_) {
      return {&weight_, &bias_};
    }
    return {&weight_};
  }

  void zero_grad() {
    weight_.zero_grad();
    if (use_bias_) {
      bias_.zero_grad();
    }
  }

  auto name() const -> std::string { return "TemporalConv1D"; }

private:
  static auto clamp_t(int t, size_t T) -> size_t {
    if (t < 0) {
      return 0;
    }
    if (static_cast<size_t>(t) >= T) {
      return T - 1;
    }
    return static_cast<size_t>(t);
  }

  void init_weights() {
    const T fan_in = static_cast<T>(in_channels_ * kernel_size_);
    const T limit = std::sqrt(T{6} / std::max(fan_in, T{1})) * T{0.1};
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<T> dist(-limit, limit);

    weight_.data.sync();
    for (size_t i = 0; i < weight_.data.size(); ++i) {
      weight_.data[i] = dist(gen);
    }
    weight_.data.sync();
    weight_.grad.zero();
    if (use_bias_) {
      bias_.data.zero();
      bias_.grad.zero();
    }
  }

  size_t in_channels_;
  size_t out_channels_;
  size_t kernel_size_;
  size_t pad_;
  bool use_bias_;
  Parameter<T> weight_;
  Parameter<T> bias_;
  Tensor<T> input_cache_;
  size_t n_{0};
  size_t t_{0};
};

} // namespace har::layers
