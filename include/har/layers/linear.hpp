#pragma once

#include "har/layers/layer.hpp"
#include "har/math.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace har::layers {

// Fully-connected layer: y = x @ W + b
// x: [batch, in_features], W: [in_features, out_features], b: [out_features]
template <std::floating_point T = float> class Linear : public Layer<T> {
public:
  Linear(size_t in_features, size_t out_features, bool bias = true)
      : in_features_(in_features), out_features_(out_features), use_bias_(bias),
        weight_({in_features, out_features}),
        bias_(bias ? Parameter<T>({out_features}, T{0}) : Parameter<T>{}) {
    init_weights();
  }

  auto forward(const Tensor<T> &input) -> Tensor<T> override {
    if (input.dims() != 2 || input.shape()[1] != in_features_) {
      throw std::invalid_argument("Linear: expected input shape [batch, " +
                                  std::to_string(in_features_) + "]");
    }

    this->input_cache_ = input;
    this->output_ = math::matmul(input, weight_.data);

    if (use_bias_) {
      const size_t batch = input.shape()[0];
      for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < out_features_; ++j) {
          this->output_.at(i, j) += bias_.data[j];
        }
      }
    }

    return this->output_;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> override {
    // dW = X^T @ dY
    weight_.grad += math::matmul(math::transpose(this->input_cache_), grad_output);

    if (use_bias_) {
      const size_t batch = grad_output.shape()[0];
      for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < out_features_; ++j) {
          bias_.grad[j] += grad_output.at(i, j);
        }
      }
    }

    // dX = dY @ W^T
    return math::matmul(grad_output, math::transpose(weight_.data));
  }

  auto name() const -> std::string override { return "Linear"; }

  auto parameters() -> std::vector<Parameter<T> *> override {
    if (use_bias_) {
      return {&weight_, &bias_};
    }
    return {&weight_};
  }

  auto weight() -> Parameter<T> & { return weight_; }
  auto bias() -> Parameter<T> & { return bias_; }

private:
  void init_weights() {
    // Xavier uniform
    const T limit =
        std::sqrt(T{6} / static_cast<T>(in_features_ + out_features_));
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<T> dist(-limit, limit);

    for (size_t i = 0; i < weight_.data.size(); ++i) {
      weight_.data[i] = dist(gen);
    }
    weight_.grad.zero();
    if (use_bias_) {
      bias_.data.zero();
      bias_.grad.zero();
    }
  }

  size_t in_features_;
  size_t out_features_;
  bool use_bias_;
  Parameter<T> weight_;
  Parameter<T> bias_;
};

} // namespace har::layers
