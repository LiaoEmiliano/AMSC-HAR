#pragma once

#include "har/layers/layer.hpp"

#include <chrono>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace har::layers {

// Inverted dropout. Identity in eval mode.
template <std::floating_point T = float> class Dropout : public Layer<T> {
public:
  explicit Dropout(T p = T{0.5}) : p_(p) {
    if (p < T{0} || p >= T{1}) {
      throw std::invalid_argument("Dropout p must be in [0, 1)");
    }
  }

  auto forward(const Tensor<T> &input) -> Tensor<T> override {
    this->input_cache_ = input;
    if (!this->training_ || p_ == T{0}) {
      this->output_ = input;
      mask_ = Tensor<T>({1}, T{1}, input.device());
      return this->output_;
    }

    this->output_ = Tensor<T>(input.shape(), input.device());
    mask_ = Tensor<T>(input.shape(), T{0}, input.device());
    const T keep = T{1} - p_;

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        const auto seed = static_cast<unsigned long long>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        cuda_ops::dropout_forward(input.data(), this->output_.data(),
                                  mask_.data(), input.size(),
                                  static_cast<float>(keep), seed);
        return this->output_;
      }
    }
#endif

    input.sync();
    const T scale = T{1} / keep;
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::bernoulli_distribution bern(static_cast<double>(keep));
    for (size_t i = 0; i < input.size(); ++i) {
      const T m = bern(gen) ? scale : T{0};
      mask_[i] = m;
      this->output_[i] = input[i] * m;
    }
    return this->output_;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> override {
    if (!this->training_ || p_ == T{0}) {
      return grad_output;
    }
    Tensor<T> grad_input(grad_output.shape(), T{0}, grad_output.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (grad_output.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::dropout_backward(grad_output.data(), mask_.data(),
                                   grad_input.data(), grad_output.size());
        return grad_input;
      }
    }
#endif

    grad_output.sync();
    mask_.sync();
    for (size_t i = 0; i < grad_output.size(); ++i) {
      grad_input[i] = grad_output[i] * mask_[i];
    }
    return grad_input;
  }

  auto name() const -> std::string override { return "Dropout"; }

private:
  T p_;
  Tensor<T> mask_;
};

} // namespace har::layers
