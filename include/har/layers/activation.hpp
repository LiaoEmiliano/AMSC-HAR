#pragma once

#include "har/layers/layer.hpp"
#include "har/tensor.hpp"

#include <algorithm>
#include <concepts>
#include <string>

namespace har::layers {

template <std::floating_point T = float> class ReLU : public Layer<T> {
public:
  auto forward(const Tensor<T> &input) -> Tensor<T> override {
    this->input_cache_ = input;
    this->output_ = Tensor<T>(input.shape());

    std::ranges::transform(input, this->output_.begin(),
                           [](T x) { return std::max(T{0}, x); });

    return this->output_;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> override {
    Tensor<T> grad_input(this->input_cache_.shape());

    for (size_t i = 0; i < grad_output.size(); ++i) {
      grad_input[i] = this->input_cache_[i] > T{0} ? grad_output[i] : T{0};
    }

    return grad_input;
  }

  auto name() const -> std::string override { return "ReLU"; }
};

} // namespace har::layers
