#pragma once

#include "har/layers/layer.hpp"

#include <stdexcept>
#include <string>

namespace har::layers {

// Flatten all dimensions after batch: [N, ...] -> [N, product(...)]
template <std::floating_point T = float> class Flatten : public Layer<T> {
public:
  auto forward(const Tensor<T> &input) -> Tensor<T> override {
    if (input.dims() < 2) {
      throw std::invalid_argument("Flatten: expected at least 2D input");
    }

    this->input_cache_ = input;
    const size_t batch = input.shape()[0];
    const size_t features = input.size() / batch;
    this->output_ = input.reshaped({batch, features});
    return this->output_;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> override {
    if (grad_output.size() != this->input_cache_.size()) {
      throw std::invalid_argument("Flatten: gradient size mismatch");
    }
    return grad_output.reshaped(this->input_cache_.shape());
  }

  auto name() const -> std::string override { return "Flatten"; }
};

} // namespace har::layers
