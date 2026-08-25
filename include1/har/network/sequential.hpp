#pragma once

#include "har/layers/layer.hpp"
#include "har/tensor.hpp"

#include <concepts>
#include <vector>

namespace har::network {

template <std::floating_point T = float> class Sequential {
public:
  Sequential() = default;

  void add(layers::LayerPtr<T> layer) { layers_.push_back(std::move(layer)); }

  auto forward(const Tensor<T> &input) -> const Tensor<T> & {
    const Tensor<T> *cur = &input;
    for (auto &layer : layers_) {
      layer->forward(*cur);
      cur = &layer->output();
    }
    return *cur;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> {
    Tensor<T> grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
      grad = (*it)->backward(grad);
    }
    return grad;
  }

  auto parameters() -> std::vector<Parameter<T> *> {
    std::vector<Parameter<T> *> params;
    for (auto &layer : layers_) {
      auto layer_params = layer->parameters();
      params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
  }

  void zero_grad() {
    for (auto &layer : layers_) {
      layer->zero_grad();
    }
  }

  void train(bool mode = true) {
    for (auto &layer : layers_) {
      layer->set_training(mode);
    }
  }

  void eval() { train(false); }

  auto layers() -> std::vector<layers::LayerPtr<T>> & { return layers_; }

  auto layers() const -> const std::vector<layers::LayerPtr<T>> & {
    return layers_;
  }

  auto num_layers() const -> size_t { return layers_.size(); }

  auto operator[](size_t idx) -> layers::Layer<T> & { return *layers_[idx]; }
  auto operator[](size_t idx) const -> const layers::Layer<T> & {
    return *layers_[idx];
  }

private:
  std::vector<layers::LayerPtr<T>> layers_;
};

} // namespace har::network
