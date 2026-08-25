#pragma once

#include "har/layers/activation.hpp"
#include "har/layers/conv2d.hpp"
#include "har/layers/dropout.hpp"
#include "har/layers/linear.hpp"
#include "har/layers/norm.hpp"
#include "har/layers/pool.hpp"
#include "har/layers/temporal_attn.hpp"
#include "har/layers/temporal_conv.hpp"
#include "har/network/sequential.hpp"
#include "har/tensor.hpp"

#include <concepts>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace har::network {

// Shared 2D CNN over video frames, then a temporal 1D conv (residual),
// then learnable temporal attention pooling.
// Input:  [N, T, C, H, W] (or [N, C, H, W] treated as T=1)
// Output: [N, num_classes]
template <std::floating_point T = float> class VideoCNN {
public:
  VideoCNN(size_t in_channels, size_t num_classes, T dropout_p = T{0.3})
      : temporal_(256, 256, 3), attn_(256), dropout_(dropout_p),
        classifier_(256, num_classes) {
    backbone_.add(std::make_unique<layers::Conv2D<T>>(in_channels, 32, 3, 1, 1));
    backbone_.add(std::make_unique<layers::BatchNorm2D<T>>(32));
    backbone_.add(std::make_unique<layers::ReLU<T>>());
    backbone_.add(std::make_unique<layers::MaxPool2D<T>>(2));
    backbone_.add(std::make_unique<layers::Conv2D<T>>(32, 64, 3, 1, 1));
    backbone_.add(std::make_unique<layers::BatchNorm2D<T>>(64));
    backbone_.add(std::make_unique<layers::ReLU<T>>());
    backbone_.add(std::make_unique<layers::MaxPool2D<T>>(2));
    backbone_.add(std::make_unique<layers::Conv2D<T>>(64, 128, 3, 1, 1));
    backbone_.add(std::make_unique<layers::BatchNorm2D<T>>(128));
    backbone_.add(std::make_unique<layers::ReLU<T>>());
    backbone_.add(std::make_unique<layers::Conv2D<T>>(128, 256, 3, 1, 1));
    backbone_.add(std::make_unique<layers::BatchNorm2D<T>>(256));
    backbone_.add(std::make_unique<layers::ReLU<T>>());
    backbone_.add(std::make_unique<layers::GlobalAvgPool2D<T>>());
  }

  auto forward(Tensor<T> input) -> Tensor<T> {
    auto &frames = flatten_time(std::move(input));
    feat_ = backbone_.forward(frames);
    timed_ = temporal_.forward(feat_, n_, t_);
    mixed_ = feat_ + timed_;
    pooled_ = attn_.forward(mixed_, n_, t_);
    auto dropped = dropout_.forward(pooled_);
    return classifier_.forward(dropped);
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> {
    auto g = classifier_.backward(grad_output);
    g = dropout_.backward(g);
    auto g_mixed = attn_.backward(g);
    auto g_timed = temporal_.backward(g_mixed);
    auto g_feat = g_mixed + g_timed;
    return restore_time(backbone_.backward(g_feat));
  }

  auto parameters() -> std::vector<Parameter<T> *> {
    auto params = backbone_.parameters();
    auto temporal_params = temporal_.parameters();
    params.insert(params.end(), temporal_params.begin(), temporal_params.end());
    auto attn_params = attn_.parameters();
    params.insert(params.end(), attn_params.begin(), attn_params.end());
    auto head = classifier_.parameters();
    params.insert(params.end(), head.begin(), head.end());
    return params;
  }

  void zero_grad() {
    backbone_.zero_grad();
    temporal_.zero_grad();
    attn_.zero_grad();
    classifier_.zero_grad();
  }

  void train(bool mode = true) {
    backbone_.train(mode);
    dropout_.set_training(mode);
    classifier_.set_training(mode);
  }

  void eval() { train(false); }

  auto name() const -> std::string { return "VideoCNN"; }

  // Learnable weights plus BatchNorm running stats, in a stable save/load order.
  auto state_tensors() -> std::vector<Tensor<T> *> {
    std::vector<Tensor<T> *> tensors;
    for (auto &layer : backbone_.layers()) {
      for (auto *param : layer->parameters()) {
        tensors.push_back(&param->data);
      }
      if (auto *bn = dynamic_cast<layers::BatchNorm2D<T> *>(layer.get())) {
        tensors.push_back(&bn->running_mean());
        tensors.push_back(&bn->running_var());
      }
    }
    for (auto *param : temporal_.parameters()) {
      tensors.push_back(&param->data);
    }
    for (auto *param : attn_.parameters()) {
      tensors.push_back(&param->data);
    }
    for (auto *param : classifier_.parameters()) {
      tensors.push_back(&param->data);
    }
    return tensors;
  }

private:
  auto flatten_time(Tensor<T> input) -> Tensor<T> & {
    frames_ = std::move(input);
    if (frames_.dims() == 5) {
      n_ = frames_.shape()[0];
      t_ = frames_.shape()[1];
      const size_t C = frames_.shape()[2];
      const size_t H = frames_.shape()[3];
      const size_t W = frames_.shape()[4];
      clip_shape_ = frames_.shape();
      frames_.reshape({n_ * t_, C, H, W});
      return frames_;
    }
    if (frames_.dims() == 4) {
      n_ = frames_.shape()[0];
      t_ = 1;
      clip_shape_ = frames_.shape();
      return frames_;
    }
    throw std::invalid_argument(
        "VideoCNN: expected [N,T,C,H,W] or [N,C,H,W] input");
  }

  auto restore_time(const Tensor<T> &grad_frames) -> Tensor<T> {
    return grad_frames.reshaped(clip_shape_);
  }

  Sequential<T> backbone_;
  layers::TemporalConv1D<T> temporal_;
  layers::TemporalAttentionPool<T> attn_;
  layers::Dropout<T> dropout_;
  layers::Linear<T> classifier_;
  Tensor<T> frames_;
  Tensor<T> feat_;
  Tensor<T> timed_;
  Tensor<T> mixed_;
  Tensor<T> pooled_;
  std::vector<size_t> clip_shape_;
  size_t n_{0};
  size_t t_{0};
};

} // namespace har::network
