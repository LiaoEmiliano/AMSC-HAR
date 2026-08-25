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

// 2D convolution over NCHW tensors.
// input:  [N, C_in, H, W]
// weight: [C_out, C_in, K_h, K_w]
// bias:   [C_out]
// output: [N, C_out, H_out, W_out]
template <std::floating_point T = float> class Conv2D : public Layer<T> {
public:
  Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size,
         size_t stride = 1, size_t padding = 0, bool bias = true)
      : Conv2D(in_channels, out_channels, kernel_size, kernel_size, stride,
               stride, padding, padding, bias) {}

  Conv2D(size_t in_channels, size_t out_channels, size_t kernel_h,
         size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
         size_t pad_w, bool bias = true)
      : in_channels_(in_channels), out_channels_(out_channels),
        kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h),
        stride_w_(stride_w), pad_h_(pad_h), pad_w_(pad_w), use_bias_(bias),
        weight_({out_channels, in_channels, kernel_h, kernel_w}),
        bias_(bias ? Parameter<T>({out_channels}, T{0}) : Parameter<T>{}) {
    init_weights();
  }

  auto forward(const Tensor<T> &input) -> Tensor<T> override {
    if (input.dims() != 4) {
      throw std::invalid_argument("Conv2D: expected NCHW input");
    }
    if (input.shape()[1] != in_channels_) {
      throw std::invalid_argument("Conv2D: in_channels mismatch");
    }

    this->input_cache_ = input;
    const size_t N = input.shape()[0];
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];
    const size_t H_out = out_dim(H, pad_h_, kernel_h_, stride_h_);
    const size_t W_out = out_dim(W, pad_w_, kernel_w_, stride_w_);

    this->output_ = Tensor<T>({N, out_channels_, H_out, W_out}, T{0},
                              input.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::conv2d_forward(
            input.data(), weight_.data.data(),
            use_bias_ ? bias_.data.data() : nullptr, this->output_.data(),
            static_cast<int>(N), static_cast<int>(in_channels_),
            static_cast<int>(H), static_cast<int>(W),
            static_cast<int>(out_channels_), static_cast<int>(kernel_h_),
            static_cast<int>(kernel_w_), static_cast<int>(stride_h_),
            static_cast<int>(stride_w_), static_cast<int>(pad_h_),
            static_cast<int>(pad_w_), static_cast<int>(H_out),
            static_cast<int>(W_out), use_bias_ ? 1 : 0);
        return this->output_;
      }
    }
#endif

    for (size_t n = 0; n < N; ++n) {
      for (size_t oc = 0; oc < out_channels_; ++oc) {
        for (size_t oh = 0; oh < H_out; ++oh) {
          for (size_t ow = 0; ow < W_out; ++ow) {
            T sum = use_bias_ ? bias_.data[oc] : T{0};
            for (size_t ic = 0; ic < in_channels_; ++ic) {
              for (size_t kh = 0; kh < kernel_h_; ++kh) {
                for (size_t kw = 0; kw < kernel_w_; ++kw) {
                  const int ih =
                      static_cast<int>(oh * stride_h_ + kh) - static_cast<int>(pad_h_);
                  const int iw =
                      static_cast<int>(ow * stride_w_ + kw) - static_cast<int>(pad_w_);
                  if (ih < 0 || iw < 0 || static_cast<size_t>(ih) >= H ||
                      static_cast<size_t>(iw) >= W) {
                    continue;
                  }
                  sum += input.at(n, ic, static_cast<size_t>(ih),
                                  static_cast<size_t>(iw)) *
                         weight_.data.at(oc, ic, kh, kw);
                }
              }
            }
            this->output_.at(n, oc, oh, ow) = sum;
          }
        }
      }
    }

    return this->output_;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> override {
    const auto &input = this->input_cache_;
    const size_t N = input.shape()[0];
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];
    const size_t H_out = grad_output.shape()[2];
    const size_t W_out = grad_output.shape()[3];

    Tensor<T> grad_input(input.shape(), T{0}, input.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::conv2d_backward(
            input.data(), weight_.data.data(), grad_output.data(),
            grad_input.data(), weight_.grad.data(),
            use_bias_ ? bias_.grad.data() : nullptr, static_cast<int>(N),
            static_cast<int>(in_channels_), static_cast<int>(H),
            static_cast<int>(W), static_cast<int>(out_channels_),
            static_cast<int>(kernel_h_), static_cast<int>(kernel_w_),
            static_cast<int>(stride_h_), static_cast<int>(stride_w_),
            static_cast<int>(pad_h_), static_cast<int>(pad_w_),
            static_cast<int>(H_out), static_cast<int>(W_out),
            use_bias_ ? 1 : 0);
        return grad_input;
      }
    }
#endif

    for (size_t n = 0; n < N; ++n) {
      for (size_t oc = 0; oc < out_channels_; ++oc) {
        for (size_t oh = 0; oh < H_out; ++oh) {
          for (size_t ow = 0; ow < W_out; ++ow) {
            const T go = grad_output.at(n, oc, oh, ow);
            if (use_bias_) {
              bias_.grad[oc] += go;
            }
            for (size_t ic = 0; ic < in_channels_; ++ic) {
              for (size_t kh = 0; kh < kernel_h_; ++kh) {
                for (size_t kw = 0; kw < kernel_w_; ++kw) {
                  const int ih =
                      static_cast<int>(oh * stride_h_ + kh) - static_cast<int>(pad_h_);
                  const int iw =
                      static_cast<int>(ow * stride_w_ + kw) - static_cast<int>(pad_w_);
                  if (ih < 0 || iw < 0 || static_cast<size_t>(ih) >= H ||
                      static_cast<size_t>(iw) >= W) {
                    continue;
                  }
                  const size_t ih_u = static_cast<size_t>(ih);
                  const size_t iw_u = static_cast<size_t>(iw);
                  weight_.grad.at(oc, ic, kh, kw) +=
                      go * input.at(n, ic, ih_u, iw_u);
                  grad_input.at(n, ic, ih_u, iw_u) +=
                      go * weight_.data.at(oc, ic, kh, kw);
                }
              }
            }
          }
        }
      }
    }

    return grad_input;
  }

  auto name() const -> std::string override { return "Conv2D"; }

  auto parameters() -> std::vector<Parameter<T> *> override {
    if (use_bias_) {
      return {&weight_, &bias_};
    }
    return {&weight_};
  }

private:
  static auto out_dim(size_t in, size_t pad, size_t kernel, size_t stride)
      -> size_t {
    if (in + 2 * pad < kernel) {
      throw std::invalid_argument("Conv2D: input too small for kernel/padding");
    }
    return (in + 2 * pad - kernel) / stride + 1;
  }

  void init_weights() {
    const T fan_in =
        static_cast<T>(in_channels_ * kernel_h_ * kernel_w_);
    // Kaiming/He uniform for ReLU: U(-sqrt(6/fan_in), sqrt(6/fan_in))
    const T limit = std::sqrt(T{6} / std::max(fan_in, T{1}));
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
  size_t kernel_h_;
  size_t kernel_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;
  bool use_bias_;
  Parameter<T> weight_;
  Parameter<T> bias_;
};

} // namespace har::layers
