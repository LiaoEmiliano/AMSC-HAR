#pragma once

#include "har/layers/layer.hpp"
#include "har/tensor.hpp"

#include <algorithm>
#include <concepts>
#include <string>
#include <type_traits>

namespace har::layers {

template <std::floating_point T = float> class ReLU : public Layer<T> {
public:
  auto forward(const Tensor<T> &input) -> Tensor<T> override {
    this->output_ = Tensor<T>(input.shape(), input.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::relu_forward(input.data(), this->output_.data(), input.size());
        return this->output_;
      }
    }
#endif

    std::transform(input.begin(), input.end(), this->output_.begin(),
                   [](T x) { return std::max(T{0}, x); });

    return this->output_;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> override {
    Tensor<T> grad_input(this->output_.shape(), this->output_.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (grad_output.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::relu_backward(this->output_.data(), grad_output.data(),
                                grad_input.data(), grad_output.size());
        return grad_input;
      }
    }
#endif

    for (size_t i = 0; i < grad_output.size(); ++i) {
      grad_input[i] = this->output_[i] > T{0} ? grad_output[i] : T{0};
    }

    return grad_input;
  }

  auto name() const -> std::string override { return "ReLU"; }
};

} // namespace har::layers
