#pragma once

#include "har/layers/layer.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace har::layers {

// Max pooling over NCHW tensors.
// input/output: [N, C, H, W] -> [N, C, H_out, W_out]
template <std::floating_point T = float> class MaxPool2D : public Layer<T> {
public:
  explicit MaxPool2D(size_t kernel_size, size_t stride = 0)
      : kernel_h_(kernel_size), kernel_w_(kernel_size),
        stride_h_(stride == 0 ? kernel_size : stride),
        stride_w_(stride == 0 ? kernel_size : stride) {}

  MaxPool2D(size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w)
      : kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h),
        stride_w_(stride_w) {}

  auto forward(const Tensor<T> &input) -> Tensor<T> override {
    if (input.dims() != 4) {
      throw std::invalid_argument("MaxPool2D: expected NCHW input");
    }

    this->input_cache_ = input;
    const size_t N = input.shape()[0];
    const size_t C = input.shape()[1];
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];

    if (H < kernel_h_ || W < kernel_w_) {
      throw std::invalid_argument("MaxPool2D: input smaller than kernel");
    }

    const size_t H_out = (H - kernel_h_) / stride_h_ + 1;
    const size_t W_out = (W - kernel_w_) / stride_w_ + 1;

    this->output_ = Tensor<T>({N, C, H_out, W_out}, input.device());
    max_indices_.assign(N * C * H_out * W_out, 0);

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        std::vector<int> idx(max_indices_.size(), 0);
        cuda_ops::maxpool_forward(
            input.data(), this->output_.data(), idx.data(), static_cast<int>(N),
            static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
            static_cast<int>(kernel_h_), static_cast<int>(kernel_w_),
            static_cast<int>(stride_h_), static_cast<int>(stride_w_),
            static_cast<int>(H_out), static_cast<int>(W_out));
        for (size_t i = 0; i < idx.size(); ++i) {
          max_indices_[i] = static_cast<size_t>(idx[i]);
        }
        return this->output_;
      }
    }
#endif

    for (size_t n = 0; n < N; ++n) {
      for (size_t c = 0; c < C; ++c) {
        for (size_t oh = 0; oh < H_out; ++oh) {
          for (size_t ow = 0; ow < W_out; ++ow) {
            T best = -std::numeric_limits<T>::infinity();
            size_t best_idx = 0;
            for (size_t kh = 0; kh < kernel_h_; ++kh) {
              for (size_t kw = 0; kw < kernel_w_; ++kw) {
                const size_t ih = oh * stride_h_ + kh;
                const size_t iw = ow * stride_w_ + kw;
                const T val = input.at(n, c, ih, iw);
                if (val > best) {
                  best = val;
                  best_idx = ih * W + iw;
                }
              }
            }
            this->output_.at(n, c, oh, ow) = best;
            max_indices_[((n * C + c) * H_out + oh) * W_out + ow] = best_idx;
          }
        }
      }
    }

    return this->output_;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> override {
    const auto &input = this->input_cache_;
    const size_t N = input.shape()[0];
    const size_t C = input.shape()[1];
    const size_t W = input.shape()[3];
    const size_t H_out = grad_output.shape()[2];
    const size_t W_out = grad_output.shape()[3];

    Tensor<T> grad_input(input.shape(), T{0}, input.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        std::vector<int> idx(max_indices_.size());
        for (size_t i = 0; i < max_indices_.size(); ++i) {
          idx[i] = static_cast<int>(max_indices_[i]);
        }
        cuda_ops::maxpool_backward(
            grad_output.data(), grad_input.data(), idx.data(),
            static_cast<int>(N), static_cast<int>(C),
            static_cast<int>(input.shape()[2]), static_cast<int>(W),
            static_cast<int>(H_out), static_cast<int>(W_out));
        return grad_input;
      }
    }
#endif

    for (size_t n = 0; n < N; ++n) {
      for (size_t c = 0; c < C; ++c) {
        for (size_t oh = 0; oh < H_out; ++oh) {
          for (size_t ow = 0; ow < W_out; ++ow) {
            const size_t flat =
                max_indices_[((n * C + c) * H_out + oh) * W_out + ow];
            const size_t ih = flat / W;
            const size_t iw = flat % W;
            grad_input.at(n, c, ih, iw) += grad_output.at(n, c, oh, ow);
          }
        }
      }
    }

    return grad_input;
  }

  auto name() const -> std::string override { return "MaxPool2D"; }

private:
  size_t kernel_h_;
  size_t kernel_w_;
  size_t stride_h_;
  size_t stride_w_;
  std::vector<size_t> max_indices_;
};

// Global average pool: [N, C, H, W] -> [N, C]
template <std::floating_point T = float> class GlobalAvgPool2D : public Layer<T> {
public:
  auto forward(const Tensor<T> &input) -> Tensor<T> override {
    if (input.dims() != 4) {
      throw std::invalid_argument("GlobalAvgPool2D: expected NCHW input");
    }
    this->input_cache_ = input;
    const size_t N = input.shape()[0];
    const size_t C = input.shape()[1];
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];
    this->output_ = Tensor<T>({N, C}, T{0}, input.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::gap_forward(input.data(), this->output_.data(),
                              static_cast<int>(N), static_cast<int>(C),
                              static_cast<int>(H), static_cast<int>(W));
        return this->output_;
      }
    }
#endif

    const T inv = T{1} / static_cast<T>(H * W);
    for (size_t n = 0; n < N; ++n) {
      for (size_t c = 0; c < C; ++c) {
        T sum = T{0};
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            sum += input.at(n, c, h, w);
          }
        }
        this->output_.at(n, c) = sum * inv;
      }
    }
    return this->output_;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> override {
    const auto &input = this->input_cache_;
    const size_t N = input.shape()[0];
    const size_t C = input.shape()[1];
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];
    Tensor<T> grad_input(input.shape(), T{0}, input.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::gap_backward(grad_output.data(), grad_input.data(),
                               static_cast<int>(N), static_cast<int>(C),
                               static_cast<int>(H), static_cast<int>(W));
        return grad_input;
      }
    }
#endif

    const T inv = T{1} / static_cast<T>(H * W);
    for (size_t n = 0; n < N; ++n) {
      for (size_t c = 0; c < C; ++c) {
        const T g = grad_output.at(n, c) * inv;
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            grad_input.at(n, c, h, w) = g;
          }
        }
      }
    }
    return grad_input;
  }

  auto name() const -> std::string override { return "GlobalAvgPool2D"; }
};

} // namespace har::layers
